# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import warp as wp

from ..core.types import Devicelike
from .articulation import eval_mass_matrix

if TYPE_CHECKING:
    from .model import Model
    from .state import State


class InverseDynamics:
    """Inverse dynamics quantities for a batch of articulated rigid-body systems."""

    class EvalType(IntEnum):
        """Bitmask flags selecting which quantities :class:`~newton.InverseDynamics` should compute.

        Flags can be combined with bitwise-or to request multiple quantities
        simultaneously; :attr:`ALL` is the union of all individual flags.
        """

        MASS_MATRIX = 1 << 0
        """Compute the joint-space mass matrix M(q)."""

        GRAVITY_COMPENSATION_FORCE = 1 << 1
        """Compute the gravity compensation generalized force G(q)."""

        CORIOLIS_COMPENSATION_FORCE = 1 << 2
        """Compute the Coriolis compensation generalized force C(q, q_dot)."""

        COMPENSATION_FORCES = GRAVITY_COMPENSATION_FORCE | CORIOLIS_COMPENSATION_FORCE
        """Compute the combined gravity and Coriolis compensation generalized forces G(q) + C(q, q_dot)."""

        ALL = MASS_MATRIX | GRAVITY_COMPENSATION_FORCE | CORIOLIS_COMPENSATION_FORCE
        """Compute the mass matrix and both compensation forces."""

    def __init__(
        self,
        max_dofs_per_articulation: int,
        articulation_count: int = 1,
        joint_dof_count: int | None = None,
        device: Devicelike | None = None,
    ):
        """Allocate output buffers for the joint-space mass matrix and compensation forces.

        The mass matrix is stored in the shape expected by
        :func:`~newton.eval_mass_matrix`:
        ``(articulation_count, max_dofs_per_articulation, max_dofs_per_articulation)``.
        ``max_dofs_per_articulation`` should match :attr:`Model.max_dofs_per_articulation`,
        which already includes any 6 DOFs contributed by a floating-base root joint.

        The compensation-force buffers share the flat joint-space layout that
        :func:`~newton.Model.joint_qd` uses. If ``joint_dof_count`` is not provided,
        it defaults to ``articulation_count * max_dofs_per_articulation`` (an upper bound).

        Args:
            max_dofs_per_articulation: Per-articulation DOF count (inclusive of
                floating-base root DOFs, if any).
            articulation_count: Number of articulations stored in the buffer.
            joint_dof_count: Total number of joint DOFs across all articulations.
            device: Warp device on which the buffers are allocated.
        """
        if joint_dof_count is None:
            joint_dof_count = articulation_count * max_dofs_per_articulation

        self.mass_matrix: wp.array3d[wp.float32] = wp.zeros(
            (articulation_count, max_dofs_per_articulation, max_dofs_per_articulation),
            dtype=wp.float32,
            device=device,
        )
        """Joint-space mass matrix M(q) [kg, kg·m, or kg·m^2, depending on the joint types of the row/column DOFs], shape (articulation_count, max_dofs_per_articulation, max_dofs_per_articulation), dtype float."""

        self.gravity_compensation_force: wp.array[wp.float32] = wp.zeros(
            joint_dof_count, dtype=wp.float32, device=device
        )
        """Generalized gravity compensation force G(q) [N or N·m, depending on joint type], shape (joint_dof_count,), dtype float."""

        self.coriolis_compensation_force: wp.array[wp.float32] = wp.zeros(
            joint_dof_count, dtype=wp.float32, device=device
        )
        """Generalized Coriolis + centrifugal compensation force C(q, q_dot) [N or N·m, depending on joint type], shape (joint_dof_count,), dtype float."""

        self.tau: wp.array[wp.float32] = wp.zeros(joint_dof_count, dtype=wp.float32, device=device)
        """Inverse-dynamics joint force ``tau = M(q)*qddot + C(q, q_dot)*q_dot + g(q)`` [N or N·m, depending on joint type], shape (joint_dof_count,), dtype float.
        Typically populated by :func:`~newton.eval_inverse_dynamics_force` from the other buffers in this container plus a user-supplied ``qddot``."""


def _rnea_compensation_pass(
    model: Model,
    state: State,
    joint_qd: wp.array[wp.float32],
    gravity: wp.array[wp.vec3],
    tau_out: wp.array[wp.float32],
) -> None:
    """Run one RNEA pass (forward + backward) and write the joint-space bias
    torque into ``tau_out``.

    With ``qdd = 0`` implicit in :func:`eval_rigid_id`, the result is the
    generalized force ``G(q)`` when ``joint_qd`` is zero (gravity only),
    ``C(q, q_dot)`` when ``gravity`` is zero (Coriolis only), or their sum
    when both are non-zero.
    """
    # Lazy import: featherstone/kernels.py imports from newton._src.sim, so a
    # top-level import here would create a circular import during sim package
    # initialization.
    from ..solvers.featherstone.kernels import (  # noqa: PLC0415
        compute_com_transforms,
        compute_spatial_inertia,
        eval_rigid_fk,
        eval_rigid_id,
        eval_rigid_tau,
    )

    device = model.device
    bc = model.body_count

    # Model-level temporaries: body-local CoM transforms and spatial inertias.
    body_X_com = wp.empty(bc, dtype=wp.transform, device=device)
    wp.launch(
        compute_com_transforms,
        dim=bc,
        inputs=[model.body_com],
        outputs=[body_X_com],
        device=device,
    )
    body_I_m = wp.empty(bc, dtype=wp.spatial_matrix, device=device)
    wp.launch(
        compute_spatial_inertia,
        dim=bc,
        inputs=[model.body_inertia, model.body_mass],
        outputs=[body_I_m],
        device=device,
    )

    # Forward kinematics into local buffers (don't mutate state.body_q).
    body_q = wp.empty_like(model.body_q)
    body_q_com = wp.empty_like(model.body_q)
    wp.launch(
        eval_rigid_fk,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            state.joint_q,
            model.joint_X_p,
            model.joint_X_c,
            body_X_com,
            model.joint_axis,
            model.joint_dof_dim,
        ],
        outputs=[body_q, body_q_com],
        device=device,
    )

    # RNEA forward pass: body bias wrenches in the spatial frame.
    dof_count = model.joint_qd.shape[0]
    joint_S_s = wp.zeros(dof_count, dtype=wp.spatial_vector, device=device)
    body_I_s = wp.zeros(bc, dtype=wp.spatial_matrix, device=device)
    body_v_s = wp.zeros(bc, dtype=wp.spatial_vector, device=device)
    body_f_s = wp.zeros(bc, dtype=wp.spatial_vector, device=device)
    body_a_s = wp.zeros(bc, dtype=wp.spatial_vector, device=device)
    wp.launch(
        eval_rigid_id,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_qd_start,
            joint_qd,
            model.joint_axis,
            model.joint_dof_dim,
            body_I_m,
            body_q,
            body_q_com,
            model.joint_X_p,
            model.body_world,
            gravity,
        ],
        outputs=[joint_S_s, body_I_s, body_v_s, body_f_s, body_a_s],
        device=device,
    )

    # RNEA backward pass: project body wrenches to joint torques. Pure
    # compensation means zero PD gains, zero limit gains, zero applied force,
    # and zero external body force — jcalc_tau collapses to -dot(S, body_f_s).
    zeros_dof = wp.zeros_like(model.joint_qd)
    zeros_body = wp.zeros(bc, dtype=wp.spatial_vector, device=device)
    body_ft_s = wp.zeros(bc, dtype=wp.spatial_vector, device=device)
    wp.launch(
        eval_rigid_tau,
        dim=model.articulation_count,
        inputs=[
            model.articulation_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_dof_dim,
            zeros_dof,  # joint_target_pos
            zeros_dof,  # joint_target_vel
            state.joint_q,
            joint_qd,
            zeros_dof,  # joint_f
            zeros_dof,  # joint_target_ke
            zeros_dof,  # joint_target_kd
            model.joint_limit_lower,
            model.joint_limit_upper,
            zeros_dof,  # joint_limit_ke
            zeros_dof,  # joint_limit_kd
            joint_S_s,
            body_f_s,
            zeros_body,  # body_f_ext
        ],
        outputs=[body_ft_s, tau_out],
        device=device,
    )


def _compute_gravity_compensation_force(
    model: Model,
    state: State,
    inverse_dynamics: InverseDynamics,
) -> None:
    """Compute G(q) into ``inverse_dynamics.gravity_compensation_force``.

    Runs RNEA with joint velocities zeroed and gravity set to
    :attr:`Model.gravity`, producing the joint-space force needed to hold the
    articulation static under gravity.
    """
    zero_qd = wp.zeros_like(model.joint_qd)
    _rnea_compensation_pass(
        model,
        state,
        zero_qd,
        model.gravity,
        inverse_dynamics.gravity_compensation_force,
    )


def _compute_coriolis_compensation_force(
    model: Model,
    state: State,
    inverse_dynamics: InverseDynamics,
) -> None:
    """Compute C(q, q_dot) into ``inverse_dynamics.coriolis_compensation_force``.

    Runs RNEA with the current joint velocities and gravity zeroed, producing
    the Coriolis + centrifugal joint-space force.
    """
    zero_gravity = wp.zeros_like(model.gravity)
    _rnea_compensation_pass(
        model,
        state,
        state.joint_qd,
        zero_gravity,
        inverse_dynamics.coriolis_compensation_force,
    )


def eval_inverse_dynamics(
    model: Model,
    state: State,
    eval_type: InverseDynamics.EvalType,
    inverse_dynamics: InverseDynamics,
) -> None:
    """Compute inverse dynamics quantities for an articulation.

    Depending on the flags in ``eval_type``, populates one or more of:
    the joint-space mass matrix M(q) [kg, kg·m, or kg·m^2, depending on the
    joint types of the row/column DOFs], the gravity compensation force G(q)
    [N or N·m, depending on joint type], and the Coriolis compensation force
    C(q, q_dot) [N or N·m, depending on joint type] into ``inverse_dynamics``.

    Args:
        model: Model providing articulation topology and inertial parameters.
        state: State providing the current generalized coordinates and velocities.
        eval_type: Bitmask selecting which quantities to compute.
        inverse_dynamics: Output container whose buffers are written in place.
    """
    if eval_type & InverseDynamics.EvalType.MASS_MATRIX:
        expected_shape = (model.articulation_count, model.max_dofs_per_articulation, model.max_dofs_per_articulation)
        if inverse_dynamics.mass_matrix.shape != expected_shape:
            raise ValueError(
                f"inverse_dynamics.mass_matrix has shape "
                f"{inverse_dynamics.mass_matrix.shape}, expected {expected_shape}."
            )
        eval_mass_matrix(model, state, H=inverse_dynamics.mass_matrix)

    if eval_type & InverseDynamics.EvalType.GRAVITY_COMPENSATION_FORCE:
        _compute_gravity_compensation_force(model, state, inverse_dynamics)

    if eval_type & InverseDynamics.EvalType.CORIOLIS_COMPENSATION_FORCE:
        _compute_coriolis_compensation_force(model, state, inverse_dynamics)
