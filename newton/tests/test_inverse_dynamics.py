# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for InverseDynamics, eval_inverse_dynamics(), and the
gravity/Coriolis compensation-force helpers."""

from __future__ import annotations

import unittest
from typing import ClassVar

import numpy as np
import warp as wp

import newton


def _gravity_vec_to_scalar_and_axis(gravity: wp.vec3) -> tuple[float, newton.Axis]:
    """Decode an axis-aligned gravity vec3 into Newton's (scalar, axis) form.

    Newton's ``ModelBuilder`` only takes scalar gravity plus an up-axis, so we
    accept at most one non-zero component and recover the signed magnitude and
    matching axis. When all components are zero, the axis is indeterminate and
    defaults to Y.
    """
    components = (float(gravity[0]), float(gravity[1]), float(gravity[2]))
    non_zero = [i for i, v in enumerate(components) if v != 0.0]
    if len(non_zero) > 1:
        raise ValueError(f"gravity must have at most one non-zero component (axis-aligned); got {components}.")
    if non_zero:
        axis_idx = non_zero[0]
        return components[axis_idx], (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[axis_idx]
    return 0.0, newton.Axis.Y


class TestInverseDynamicsBase:
    """Shared test body. Concrete subclasses set :attr:`device`."""

    device: wp.context.Device | None = None

    # Per-link inertia tensors swept by tests in :class:`TestGravCompForce`
    # to confirm G(q) is genuinely insensitive to the inertia tensor (it
    # depends only on mass and CoM); also used as default inertias in the
    # other test classes to avoid repeating the identity-inertia literal.
    I_UNIT: ClassVar[wp.mat33] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    I_100: ClassVar[wp.mat33] = wp.mat33(100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0)
    INERTIA_PASSES: ClassVar[list[wp.mat33]] = [I_UNIT, I_100]

    @staticmethod
    def _build_two_link_pendulum(
        gravity: wp.vec3,
        floating_base: bool,
        joint_type: str,
        joint_axis: wp.vec3,
        link_coms: list[wp.vec3],
        link_masses: list[float],
        joint_frames: list[wp.transform],
        link_inertias: list[wp.mat33],
    ) -> newton.ModelBuilder:
        gravity_scalar, up_axis = _gravity_vec_to_scalar_and_axis(gravity)
        builder = newton.ModelBuilder(gravity=gravity_scalar, up_axis=up_axis)

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        if joint_type == "revolute":
            add_dof_joint = builder.add_joint_revolute
        elif joint_type == "prismatic":
            add_dof_joint = builder.add_joint_prismatic
        else:
            raise ValueError(f"joint_type must be 'revolute' or 'prismatic', got {joint_type!r}.")

        b1 = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=link_masses[0],
            inertia=link_inertias[0],
            com=link_coms[0],
        )
        if floating_base:
            j1 = builder.add_joint_free(
                parent=-1,
                child=b1,
                parent_xform=identity_xform,
                child_xform=identity_xform,
            )
        else:
            j1 = builder.add_joint_fixed(
                parent=-1,
                child=b1,
                parent_xform=identity_xform,
                child_xform=identity_xform,
            )

        b2 = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=link_masses[1],
            inertia=link_inertias[1],
            com=link_coms[1],
        )
        j2 = add_dof_joint(
            parent=b1,
            child=b2,
            axis=joint_axis,
            parent_xform=joint_frames[0],
            child_xform=joint_frames[1],
        )
        builder.add_articulation([j1, j2], label="pendulum")

        return builder

    def _pose_pendulum(self, model):
        state = model.state()
        joint_q = state.joint_q.numpy()
        # Apply a non-trivial pose to whichever coords exist. Fixed-root + one DOF
        # has joint_q of length 1; floating root (7 free coords) + one DOF has
        # length 8 — both cases get at least the leading entry perturbed.
        if joint_q.shape[0] >= 1:
            joint_q[0] = 0.3
        if joint_q.shape[0] >= 2:
            joint_q[1] = 0.5
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        return state


class TestManipulatorEquation(TestInverseDynamicsBase):
    """Manipulator-equation tests covering combined inverse-dynamics outputs."""

    def test_eval_all_populates_every_buffer(self):
        """EvalType.ALL must write the mass matrix and both compensation forces in one call.

        Uses a floating base so the articulation has multi-DOF coupling; a
        fixed root with a single revolute DOF has identically-zero Coriolis
        and would trivially defeat that assertion.
        """
        builder = self._build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=True,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)
        joint_qd = state.joint_qd.numpy()
        # Populate linear, angular, and internal DOFs so Coriolis has real
        # coupling: pure linear base motion doesn't couple through C(q, q_dot).
        joint_qd[:] = np.linspace(0.1, 0.7, joint_qd.shape[0])
        state.joint_qd.assign(joint_qd)

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(model, state, newton.InverseDynamics.EvalType.ALL, inverse_dynamics)

        H = inverse_dynamics.mass_matrix.numpy()
        g = inverse_dynamics.gravity_compensation_force.numpy()
        c = inverse_dynamics.coriolis_compensation_force.numpy()

        self.assertTrue(np.all(np.isfinite(H)))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertGreater(float(np.linalg.norm(H)), 1e-6)
        self.assertGreater(float(np.linalg.norm(g)), 1e-6)
        self.assertGreater(float(np.linalg.norm(c)), 1e-6)


class TestGravCompForce(TestInverseDynamicsBase):
    """Gravity-compensation-force tests for the two-link pendulum harness."""

    @staticmethod
    def _default_joint_q(is_floating_base: list[list[bool]]) -> list[list[list[float]]]:
        """Build the default initial-state ``joint_q`` for a multi-world,
        multi-articulation pendulum: zero position, identity quaternion,
        zero internal q for each floating articulation; a single zero
        internal q for each fixed one.
        """
        default_floating = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        default_fixed = [0.0]
        return [
            [list(default_floating) if floating else list(default_fixed) for floating in row]
            for row in is_floating_base
        ]

    def _test_two_link_grav_comp_force(
        self,
        gravity_vec: wp.vec3,
        joint_type: str,
        is_floating_base: list[list[bool]],
        joint_axis: list[list[wp.vec3]],
        joint_frames: list[list[list[wp.transform]]],
        joint_q: list[list[list[float]]],
        link_coms: list[list[list[wp.vec3]]],
        link_masses: list[list[list[float]]],
        link_inertias: list[list[list[wp.mat33]]],
        expected_grav_comp_forces: list[float],
    ):
        """G(q) is populated correctly for a multi-world, multi-articulation model.

        Args:
            gravity_vec: Axis-aligned gravity (world frame).
            joint_type: ``"revolute"`` or ``"prismatic"`` — shared across every articulation.
            is_floating_base: Per-articulation floating-vs-fixed root flag ``[w][a]``.
            joint_axis: Per-articulation joint axis ``[w][a]`` (same shape as
                ``is_floating_base``).
            joint_frames: Per-joint parent-side anchor transforms ``[w][a][joint]``.
                ``joint[0]`` is the root joint (free or fixed), ``joint[1]`` is
                the internal DOF joint.
            joint_q: Per-articulation initial-state ``joint_q`` shaped
                ``[w][a]``. Each inner list holds the per-articulation
                generalized coordinates: 8 floats for a floating root with
                one internal DOF (3 base position, 4 base quaternion, 1
                internal q) and 1 float for a fixed root with one internal
                DOF (``(q_internal,)``). Written into ``state.joint_q``
                before ``eval_fk`` so the rest pose used during
                gravity-compensation evaluation reflects this input. Use
                :meth:`_default_joint_q` to build the zero-pose /
                identity-quat default when the test doesn't care about the
                rest pose.
            link_coms: Per-link CoM offsets ``[w][a][link]`` as ``wp.vec3``.
            link_masses: Per-link masses ``[w][a][link]``.
            link_inertias: Per-link body-frame inertia tensors
                ``[w][a][link]`` as ``wp.mat33``.
            expected_grav_comp_forces: Flat expected ``-G(q)`` in the order
                Newton reports them.
        """
        gravity_scalar, up_axis = _gravity_vec_to_scalar_and_axis(gravity_vec)

        # Derive shape constants from the structured inputs.
        num_worlds = len(is_floating_base)
        num_arts_per_world = len(is_floating_base[0])
        num_links_per_articulation = len(link_coms[0][0])
        # _build_two_link_pendulum hard-codes a two-link articulation, so the
        # caller's link_coms layout must agree.
        self.assertEqual(num_links_per_articulation, 2)

        # Each articulation contributes 7 DOFs if floating (6 free-joint
        # DOFs + 1 internal) or 1 DOF if fixed. G(q) and
        # expected_grav_comp_forces are sized by total DOF count.
        expected_total_dofs = sum(7 if floating else 1 for row in is_floating_base for floating in row)
        if len(expected_grav_comp_forces) != expected_total_dofs:
            raise ValueError(
                f"expected_grav_comp_forces has length {len(expected_grav_comp_forces)}, "
                f"but is_floating_base implies {expected_total_dofs} total DOFs."
            )

        # Build the model from the structured per-world / per-articulation inputs.
        model_builder = newton.ModelBuilder(gravity=gravity_scalar, up_axis=up_axis)
        for i in range(0, num_worlds):
            world_builder = newton.ModelBuilder(gravity=gravity_scalar, up_axis=up_axis)
            for j in range(0, num_arts_per_world):
                articulation_builder = self._build_two_link_pendulum(
                    gravity=gravity_vec,
                    joint_type=joint_type,
                    joint_axis=joint_axis[i][j],
                    floating_base=is_floating_base[i][j],
                    link_coms=link_coms[i][j],
                    link_masses=link_masses[i][j],
                    joint_frames=joint_frames[i][j],
                    link_inertias=link_inertias[i][j],
                )
                world_builder.add_builder(articulation_builder)
            model_builder.add_world(world_builder)

        model = model_builder.finalize(device=self.device)
        state = model.state()

        # Patch the per-articulation joint_q ranges in the global state
        # vector. Articulations are appended in (world, articulation) iteration
        # order by the build loop above, and within an articulation the root
        # joint comes first, so the q layout is 7 free-joint values (3 base
        # position + 4 base quaternion) + 1 internal DOF for floating roots,
        # or just 1 internal DOF for fixed roots.
        joint_q_arr = state.joint_q.numpy()
        offset = 0
        for i in range(num_worlds):
            for j in range(num_arts_per_world):
                art_q_size = 8 if is_floating_base[i][j] else 1
                override = joint_q[i][j]
                if len(override) != art_q_size:
                    raise ValueError(
                        f"joint_q[{i}][{j}] has length {len(override)}, "
                        f"expected {art_q_size} for is_floating_base={is_floating_base[i][j]}."
                    )
                joint_q_arr[offset : offset + art_q_size] = override
                offset += art_q_size
        state.joint_q.assign(joint_q_arr)

        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        inverse_dynamics = model.inverse_dynamics()

        newton.eval_inverse_dynamics(
            model=model,
            state=state,
            eval_type=newton.InverseDynamics.EvalType.GRAVITY_COMPENSATION_FORCE,
            inverse_dynamics=inverse_dynamics,
        )

        measured_gravity_comp_force = inverse_dynamics.gravity_compensation_force.numpy()
        self.assertTrue(np.all(np.isfinite(measured_gravity_comp_force)))

        # Newton's gravity_compensation_force returns G(q); the force the user
        # would apply to hold the articulation static is -G(q), which is what
        # expected_grav_comp_forces lists, so compare against -tau.
        self.assertEqual(measured_gravity_comp_force.shape, (len(expected_grav_comp_forces),))
        np.testing.assert_allclose(-measured_gravity_comp_force, expected_grav_comp_forces, atol=1e-5, rtol=1e-5)

    def test_gravity_zero_without_gravity(self):
        """G(q) must vanish when the model has zero gravity."""
        for I in self.INERTIA_PASSES:
            builder = self._build_two_link_pendulum(
                gravity=wp.vec3(0.0, 0.0, 0.0),
                floating_base=False,
                joint_type="revolute",
                joint_axis=wp.vec3(0.0, 0.0, 1.0),
                link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
                link_masses=[1.0, 2.0],
                joint_frames=[
                    wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                    wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                ],
                link_inertias=[I, I],
            )
            model = builder.finalize(device=self.device)
            state = self._pose_pendulum(model)

            inverse_dynamics = model.inverse_dynamics()
            newton.eval_inverse_dynamics(
                model,
                state,
                newton.InverseDynamics.EvalType.GRAVITY_COMPENSATION_FORCE,
                inverse_dynamics,
            )

            tau = inverse_dynamics.gravity_compensation_force.numpy()
            np.testing.assert_allclose(tau, np.zeros_like(tau), atol=1e-6)

    def test_gravity_nonzero_under_gravity(self):
        """G(q) is generically non-zero for a non-trivial pose under gravity.

        Link 1's body-frame CoM is offset to (1, 0, 0) so that after the
        revolute-about-z rotation the distal mass sits off the joint axis;
        without such a lever arm G(q) would be identically zero and the
        ``|tau| > 1e-6`` assertion could not distinguish a correct solver
        from one that returns zeros.
        """
        for I in self.INERTIA_PASSES:
            builder = self._build_two_link_pendulum(
                gravity=wp.vec3(0.0, -9.81, 0.0),
                floating_base=False,
                joint_type="revolute",
                joint_axis=wp.vec3(0.0, 0.0, 1.0),
                link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0)],
                link_masses=[1.0, 2.0],
                joint_frames=[
                    wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                    wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                ],
                link_inertias=[I, I],
            )
            model = builder.finalize(device=self.device)
            state = self._pose_pendulum(model)

            inverse_dynamics = model.inverse_dynamics()
            newton.eval_inverse_dynamics(
                model,
                state,
                newton.InverseDynamics.EvalType.GRAVITY_COMPENSATION_FORCE,
                inverse_dynamics,
            )

            tau = inverse_dynamics.gravity_compensation_force.numpy()
            self.assertGreater(float(np.linalg.norm(tau)), 1e-6)

    def test_two_link_grav_comp_force_from_zero_gravity(self):
        """G(q) vanishes everywhere when the model has zero gravity.

        Builds a multi-world, multi-articulation pendulum (2 worlds x 2
        articulations, mixed fixed/floating roots) for each of the supported
        internal joint types (revolute, prismatic) with per-articulation joint
        axes. With gravity set to the zero vector, the generalized gravity
        force must be identically zero on every DOF, independent of joint
        type, joint axis, root type, link mass, CoM offset, or link inertia
        tensor. The outer loop runs once with unit inertias and once with
        inertias scaled by 100 to confirm G(q) is truly insensitive to the
        inertia tensor.
        """
        joint_types = ["revolute", "prismatic"]

        gravity_vec = wp.vec3(0.0, 0.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_x = wp.vec3(1.0, 0.0, 0.0)
        prismatic_y = wp.vec3(0.0, 1.0, 0.0)
        prismatic_z = wp.vec3(0.0, 0.0, 1.0)
        joint_axis = [
            [prismatic_x, prismatic_y],  # World0, articulation0/articulation1
            [prismatic_z, prismatic_x],  # World1, articulation0/articulation1
        ]

        # Non-identity anchors on the internal joint — under zero gravity
        # G(q) must still be identically zero, independent of where the
        # joint is anchored in the parent/child bodies or how either frame
        # is oriented. The hand-written quaternion values below encode
        # 45 deg about +z and 60 deg about +y.
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        shift_x = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        shift_ny = wp.transform(wp.vec3(0.0, -0.3, 0.0), wp.quat_identity())
        shift_z = wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity())
        rot_z_45 = wp.transform(wp.vec3(0.1, 0.2, 0.0), wp.quat(0.0, 0.0, 0.3826834, 0.9238795))
        rot_y_60 = wp.transform(wp.vec3(0.0, 0.0, -0.4), wp.quat(0.0, 0.5, 0.0, 0.8660254))
        joint_frames = [
            [
                [shift_x, identity_xform],  # World0, articulation0, internal joint parent/child xforms
                [shift_ny, rot_y_60],  # World0, articulation1, internal joint parent/child xforms
            ],
            [
                [rot_z_45, shift_z],  # World1, articulation0, internal joint parent/child xforms
                [shift_x, rot_z_45],  # World1, articulation1, internal joint parent/child xforms
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, -2.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0, -1.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 3.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0,
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        expected_grav_comp_forces = [
            0.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],  # World0, articulation0/articulation1, link0/link1
                [[I, I], [I, I]],  # World1, articulation0/articulation1, link0/link1
            ]
            for i in range(0, 2):
                self._test_two_link_grav_comp_force(
                    gravity_vec=gravity_vec,
                    joint_type=joint_types[i],
                    is_floating_base=is_floating_base,
                    joint_axis=joint_axis,
                    joint_frames=joint_frames,
                    joint_q=joint_q,
                    link_coms=link_coms,
                    link_masses=link_masses,
                    link_inertias=link_inertias,
                    expected_grav_comp_forces=expected_grav_comp_forces,
                )

    def test_two_link_prismatic_grav_comp_force_from_mass(self):
        """A prismatic DOF aligned with gravity carries ``G(q) = m_distal * g``.

        With gravity along -y and the internal prismatic axis along +y
        (fully aligned), zero CoMs, identity joint frames, and zero
        internal q, each articulation's internal DOF carries
        ``m_distal * |g|`` — the parent link is reacted by either the
        fixed root or the floating base and so contributes nothing on
        the internal slider. Floating-root articulations additionally
        carry ``M_total * |g|`` on the base linear-y entry; angular and
        the other linear base entries are zero (no lever arm with zero
        CoMs). The four articulations sweep distal masses 2, 4, 6, 8 to
        confirm the slider entry scales linearly with ``m_distal``.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_y = wp.vec3(0.0, 1.0, 0.0)
        joint_axis = [
            [prismatic_y, prismatic_y],  # World0, articulation0/articulation1
            [prismatic_y, prismatic_y],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World0, articulation1, root/internal joint
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World1, articulation1, root/internal joint
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0,
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        expected_grav_comp_forces = [
            20.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs
            70.0,
            0.0,
            0.0,
            0.0,
            0.0,
            40.0,
            60.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            150.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_prismatic_grav_comp_force_from_rotated_root(self):
        """Body +X slider + per-articulation +/- 90 deg root rotation about
        +Z, applied through the free joint's quaternion in ``joint_q``.

        Every articulation here is floating-base so the rotation can live in
        ``joint_q``. World 0 a0 and World 1 a1 use a +90 deg rotation
        (body +X maps to world +Y, prismatic-DOF entry of ``-G(q)`` is
        ``+m_2 * g``); World 0 a1 and World 1 a0 use a -90 deg rotation
        (body +X maps to world -Y, prismatic-DOF entry is ``-m_2 * g``).
        With all CoMs zero the per-articulation expected ``-G(q)`` is
        ``(0, M_total * g, 0, 0, 0, 0, +/- m_2 * g)``, with the prismatic
        sign matching the sign of the rotation.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [True, True],  # World0, all floating so joint_q can rotate every root
            [True, True],  # World1
        ]

        prismatic_x = wp.vec3(1.0, 0.0, 0.0)
        joint_axis = [
            [prismatic_x, prismatic_x],  # World0, articulation0/articulation1
            [prismatic_x, prismatic_x],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World0, articulation1, root/internal joint
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World1, articulation1, root/internal joint
            ],
        ]

        # Build two per-articulation joint_q lists encoding +/- 90 deg
        # rotations about +Z, zero base position, and zero internal q.
        root_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        floating_q_rot_z_90 = [0.0, 0.0, 0.0, root_quat.x, root_quat.y, root_quat.z, root_quat.w, 0.0]

        root_quat_neg = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0)
        floating_q_rot_z_neg90 = [
            0.0,
            0.0,
            0.0,
            root_quat_neg.x,
            root_quat_neg.y,
            root_quat_neg.z,
            root_quat_neg.w,
            0.0,
        ]

        joint_q = [
            [floating_q_rot_z_90, floating_q_rot_z_neg90],  # World0: a0 = +90 deg, a1 = -90 deg
            [floating_q_rot_z_neg90, floating_q_rot_z_90],  # World1: a0 = -90 deg, a1 = +90 deg
        ]

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        # Per articulation: 6 base DOFs + 1 internal = 7 DOFs.
        # lin_x = lin_z = 0 (gravity has no x/z component);
        # lin_y = M_total * 10 (total weight);
        # ang_xyz = 0 (zero CoMs -> no lever arm);
        # prismatic = +/- m_2 * 10: +Y world-slider gives +m_2*10, the -90 deg
        # rotation flips the slider to -Y and flips the sign for those rows.
        # World0 a1 and World1 a0 use floating_q_rot_z_neg90; the other two
        # use floating_q_rot_z_90.
        expected_grav_comp_forces = [
            0.0,
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            20.0,  # W0 a0 [1, 2]: M=3, m_2=2  (+90 deg)
            0.0,
            70.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -40.0,  # W0 a1 [3, 4]: M=7, m_2=4  (-90 deg)
            0.0,
            110.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -60.0,  # W1 a0 [5, 6]: M=11, m_2=6 (-90 deg)
            0.0,
            150.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,  # W1 a1 [7, 8]: M=15, m_2=8 (+90 deg)
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_prismatic_grav_comp_force_from_rotated_joint_frame(self):
        """Body +X slider with per-articulation +/- 90 deg rotation of the
        internal joint frame about +Z, instead of rotating the floating root.

        This sibling of
        :meth:`test_two_link_prismatic_grav_comp_force_from_rotated_root`
        achieves the world-frame slider direction by rotating the *internal
        joint's ``parent_xform``* about +Z. The roots themselves stay at
        identity orientation, and unlike the rotated-root variant we don't
        need the roots to be free for the joint-frame rotation to take
        effect — so this test mixes fixed and floating roots.

        Per articulation, with zero CoMs:
          - +90 deg joint-frame rotation maps body +X to world +Y, so the
            prismatic-DOF entry is ``+m_2 * g``;
          - -90 deg joint-frame rotation maps body +X to world -Y, so the
            prismatic-DOF entry is ``-m_2 * g``.
          - Floating-root articulations additionally carry the base linear-y
            entry ``M_total * g``; angular and the other linear entries are
            zero (no lever arm with zero CoMs).
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_x = wp.vec3(1.0, 0.0, 0.0)
        joint_axis = [
            [prismatic_x, prismatic_x],  # World0, articulation0/articulation1
            [prismatic_x, prismatic_x],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

        # Joint-frame quaternion: pi/2 rotation about +Z. After parent_xform
        # is applied to the parent body's frame, the joint's local frame is
        # rotated 90 deg about Z, so the joint axis (1, 0, 0) in local
        # coords corresponds to (0, 1, 0) in the parent body frame — i.e.
        # +Y in the world when the root is at identity.
        joint_frame_quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2.0)
        parent_xform_rot_z_90 = wp.transform(wp.vec3(0.0, 0.0, 0.0), joint_frame_quat)

        # Counter-rotated joint frame: -pi/2 about +Z maps body +X to world -Y.
        joint_frame_quat_neg = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -np.pi / 2.0)
        parent_xform_rot_z_neg90 = wp.transform(wp.vec3(0.0, 0.0, 0.0), joint_frame_quat_neg)

        # joint_frames[w][a] = [parent_xform_internal, child_xform_internal].
        # +90 deg parent_xform maps body +X to world +Y; -90 deg maps body
        # +X to world -Y, used here on World0 a1 and World1 a0.
        joint_frames = [
            [
                [parent_xform_rot_z_90, identity_xform],  # World0, articulation0 (+90 deg)
                [parent_xform_rot_z_neg90, identity_xform],  # World0, articulation1 (-90 deg)
            ],
            [
                [parent_xform_rot_z_neg90, identity_xform],  # World1, articulation0 (-90 deg)
                [parent_xform_rot_z_90, identity_xform],  # World1, articulation1 (+90 deg)
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        # Fixed root: 1 DOF (internal prismatic only).
        # Floating root: 6 base DOFs + 1 internal = 7 DOFs.
        # prismatic = +/- m_2 * 10 (sign matches the joint-frame rotation).
        # For floating articulations: lin_y = M_total * 10 (total weight),
        # all other base entries = 0 (zero CoMs -> no lever arm).
        expected_grav_comp_forces = [
            20.0,  # W0 a0 [1, 2]: fixed,    +90 deg, m_2=2
            0.0,
            70.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -40.0,  # W0 a1 [3, 4]: floating, -90 deg, M=7, m_2=4
            -60.0,  # W1 a0 [5, 6]: fixed,    -90 deg, m_2=6
            0.0,
            150.0,
            0.0,
            0.0,
            0.0,
            0.0,
            80.0,  # W1 a1 [7, 8]: floating, +90 deg, M=15, m_2=8
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_prismatic_grav_comp_force_from_com(self):
        """Lateral CoM offsets produce angular-z entries on floating roots.

        Same setup as
        :meth:`test_two_link_prismatic_grav_comp_force_from_mass` —
        prismatic +y axis aligned with -y gravity, identity joint frames,
        zero internal q — but with non-zero per-link CoM offsets along
        +x. The internal prismatic entry is unchanged (a transverse lever
        arm doesn't project onto an axial slider DOF), and the linear-y
        base entry on floating roots remains ``M_total * |g|``. Floating
        roots additionally develop an angular-z entry equal to
        ``sum_i m_i * x_i * |g|`` from the cross product
        ``r_com x (0, -g, 0)``; angular x and y stay zero because the
        CoM offsets have no y or z component.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        prismatic_y = wp.vec3(0.0, 1.0, 0.0)
        joint_axis = [
            [prismatic_y, prismatic_y],  # World0, articulation0/articulation1
            [prismatic_y, prismatic_y],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World0, articulation1, root/internal joint
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, root/internal joint
                [identity_xform, identity_xform],  # World1, articulation1, root/internal joint
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.5, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [3.0, 4.0]],  # World0,
            [[5.0, 6.0], [7.0, 8.0]],  # World1
        ]

        expected_grav_comp_forces = [
            20.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs
            70.0,
            0.0,
            0.0,
            0.0,
            15.0,
            40.0,
            60.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            150.0,
            0.0,
            0.0,
            0.0,
            75.0,
            80.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_prismatic_grav_comp_force_axis_perpendicular_to_gravity(self):
        """A prismatic DOF whose axis is perpendicular to gravity carries zero G(q).

        With gravity along -y and the internal prismatic axis along +x, the
        projection of the gravity force on the joint axis is ``g . axis = 0``,
        so the generalized force on the internal DOF vanishes regardless of
        the distal link's mass. The per-articulation internal-DOF entry in
        G(q) must be exactly zero. On the floating-root articulations the
        base's linear-y entry still picks up the total weight, which acts as
        a sanity check that gravity is actually being applied.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        # Prismatic axis perpendicular to gravity: zero projection on the axis.
        prismatic_x = wp.vec3(1.0, 0.0, 0.0)
        joint_axis = [
            [prismatic_x, prismatic_x],  # World0, articulation0/articulation1
            [prismatic_x, prismatic_x],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, internal joint parent/child xforms
                [identity_xform, identity_xform],  # World0, articulation1, internal joint parent/child xforms
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, internal joint parent/child xforms
                [identity_xform, identity_xform],  # World1, articulation1, internal joint parent/child xforms
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        # Zero CoMs and identity joint anchors keep the non-internal entries
        # analytically tractable; the invariant under test is the internal
        # DOF entry, which is zero independent of these choices.
        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [1.0, 2.0]],  # World0
            [[1.0, 2.0], [1.0, 2.0]],  # World1
        ]

        # Fixed root: only DOF is the internal prismatic -> 0 (invariant).
        # Floating root: (v_x, v_y, v_z, omega_x, omega_y, omega_z, q_internal).
        #   Linear y = M_total * |g| = 3 * 10 = 30 (total weight).
        #   Angular and internal = 0 (both CoMs at root origin; axis perp to gravity).
        expected_grav_comp_forces = [
            0.0,  # World 0, fixed root, 1 dof (internal prismatic)
            0.0,  # World 0, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # World 1, fixed root, 1 dof (internal prismatic)
            0.0,  # World 1, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="prismatic",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_revolute_grav_comp_force_from_jnt_frame(self):
        """Sweeps the internal-joint ``child_xform`` to verify ``G(q)`` for
        a revolute DOF tracks the moment arm of the distal link.

        Each articulation has a revolute-about-+z internal joint with zero
        body CoMs and identity ``parent_xform``. The per-articulation
        ``child_xform`` translation (and one ``+y`` rotation) displaces the
        distal link's origin — and therefore its CoM, since the body CoM
        is zero — to a known world position at zero internal q. With
        gravity along ``-y`` and revolute axis ``+z``, the internal-DOF
        entry of ``-G(q)`` reduces to ``m_distal * |g| * x_world``.
        Articulations whose displacement is along ``+/- y`` or ``+/- z``
        therefore have a zero internal entry. Floating-root articulations
        additionally carry ``M_total * |g|`` on the base linear-y entry
        and ``r_com x (0, -g, 0)`` on the angular entries — confirming the
        solver correctly picks up the joint-frame placement (translation
        and rotation) on every block of the floating base, not just on
        the internal DOF. Concretely:

        - W0 a0 (fixed, child = (-4, 0, 0) identity): CoM at (4, 0, 0),
          internal entry = ``2 * 10 * 4 = 80``.
        - W0 a1 (floating, child = (0, -4, 0) rotated 90 deg about +y):
          CoM at (0, 4, 0), internal entry zero, base linear-y = 30.
        - W1 a0 (fixed, child = (0, -4, 0) identity): CoM at (0, 4, 0),
          internal entry zero (CoM offset parallel to gravity).
        - W1 a1 (floating, child = (0, 0, -4) identity): CoM at
          (0, 0, 4), base angular-x = ``-80``, base linear-y = 30,
          internal entry zero.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        revolute_z = wp.vec3(0.0, 0.0, 1.0)
        joint_axis = [
            [revolute_z, revolute_z],  # World0, articulation0/articulation1
            [revolute_z, revolute_z],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        child_anchor_0 = wp.transform(wp.vec3(-4.0, 0.0, 0.0), wp.quat_identity())
        child_anchor_1 = wp.transform(wp.vec3(0.0, -4.0, 0.0), wp.quat(0.0, 0.7071068, 0.0, 0.7071068))
        child_anchor_2 = wp.transform(wp.vec3(0.0, -4.0, 0.0), wp.quat_identity())
        child_anchor_3 = wp.transform(wp.vec3(0.0, 0.0, -4.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, child_anchor_0],  # World0, articulation0, internal joint parent/child xforms
                [identity_xform, child_anchor_1],  # World0, articulation1, internal joint parent/child xforms
            ],
            [
                [identity_xform, child_anchor_2],  # World1, articulation0, internal joint parent/child xforms
                [identity_xform, child_anchor_3],  # World1, articulation1, internal joint parent/child xforms
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [1.0, 2.0]],  # World0,
            [[1.0, 2.0], [1.0, 2.0]],  # World1
        ]

        expected_grav_comp_forces = [
            80.0,  # World 0, fixed root, 1 dof
            0.0,  # World 0, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # World 1, fixed root, 1 dof
            0.0,  # World 1, floating root, 6+1 dofs
            30.0,
            0.0,
            -80.0,
            0.0,
            0.0,
            0.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="revolute",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_revolute_grav_comp_force_axis_parallel_to_gravity(self):
        """A revolute DOF whose axis is parallel to gravity carries zero G(q).

        With gravity along -y and the internal revolute axis along +y, the
        gravity force on the distal link is always collinear with the joint
        axis, so the moment ``(r x F).axis`` is identically zero for any
        lever arm ``r``. The per-articulation internal-DOF entry in G(q)
        must therefore be exactly zero, regardless of CoM, mass, joint
        anchor, or base pose. On the floating-root articulations, the base's
        linear-y entry still picks up the total weight, which acts as a
        sanity check that gravity is actually being applied.
        """
        gravity_vec = wp.vec3(0.0, -10.0, 0.0)

        is_floating_base = [
            [False, True],  # World0, articulation0 fixed, articulation1 free
            [False, True],  # World1, articulation0 fixed, articulation1 free
        ]

        # Revolute axis aligned with gravity: zero moment about the axis.
        revolute_y = wp.vec3(0.0, 1.0, 0.0)
        joint_axis = [
            [revolute_y, revolute_y],  # World0, articulation0/articulation1
            [revolute_y, revolute_y],  # World1, articulation0/articulation1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform],  # World0, articulation0, internal joint parent/child xforms
                [identity_xform, identity_xform],  # World0, articulation1, internal joint parent/child xforms
            ],
            [
                [identity_xform, identity_xform],  # World1, articulation0, internal joint parent/child xforms
                [identity_xform, identity_xform],  # World1, articulation1, internal joint parent/child xforms
            ],
        ]

        joint_q = self._default_joint_q(is_floating_base)

        # Zero CoMs and identity joint anchors keep the non-internal entries
        # analytically tractable; the invariant under test is the internal
        # DOF entry, which is zero independent of these choices.
        link_coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World0, articulation1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],  # World1, articulation1, link0/link1
            ],
        ]
        link_masses = [
            [[1.0, 2.0], [1.0, 2.0]],  # World0
            [[1.0, 2.0], [1.0, 2.0]],  # World1
        ]

        # Fixed root: only DOF is the internal revolute -> 0 (invariant).
        # Floating root: (v_x, v_y, v_z, omega_x, omega_y, omega_z, q_internal).
        #   Linear y = M_total * |g| = 3 * 10 = 30 (total weight).
        #   Angular and internal = 0 (both CoMs at root origin; axis ∥ gravity).
        expected_grav_comp_forces = [
            0.0,  # World 0, fixed root, 1 dof (internal revolute)
            0.0,  # World 0, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # World 1, fixed root, 1 dof (internal revolute)
            0.0,  # World 1, floating root, 6+1 dofs
            30.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        for I in self.INERTIA_PASSES:
            link_inertias = [
                [[I, I], [I, I]],
                [[I, I], [I, I]],
            ]
            self._test_two_link_grav_comp_force(
                gravity_vec=gravity_vec,
                joint_type="revolute",
                is_floating_base=is_floating_base,
                joint_axis=joint_axis,
                joint_frames=joint_frames,
                joint_q=joint_q,
                link_coms=link_coms,
                link_masses=link_masses,
                link_inertias=link_inertias,
                expected_grav_comp_forces=expected_grav_comp_forces,
            )

    def test_two_link_fixed_revolute_gravity_compensation_matches_closed_form(self):
        """Internal revolute DOF matches the closed-form ``m * g * arm_length * cos(q)``.

        Fixed-root arm, internal revolute about +z anchored at the world
        origin. The child body's origin is placed at ``(arm_length, 0, 0)``
        by setting the internal joint's ``child_xform`` to
        ``(-arm_length, 0, 0)``; the distal link's body-frame CoM is the
        origin, so link 1's world CoM at angle ``q`` is
        ``(arm_length * cos q, arm_length * sin q, 0)``. Under gravity
        ``(0, -g, 0)`` the Lagrangian generalized gravity force on the
        internal DOF reduces to
        ``G(q) = m_distal * g * arm_length * cos(q)``. Newton returns
        ``-G(q)`` with the sign convention used elsewhere in this file, so
        we compare ``-tau[0]`` against the closed form over a pose sweep.
        """
        arm_length = 1.0
        m_distal = 2.0
        g_mag = 10.0

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        child_anchor_back = wp.transform(wp.vec3(-arm_length, 0.0, 0.0), wp.quat_identity())
        for I in self.INERTIA_PASSES:
            builder = self._build_two_link_pendulum(
                gravity=wp.vec3(0.0, -g_mag, 0.0),
                floating_base=False,
                joint_type="revolute",
                joint_axis=wp.vec3(0.0, 0.0, 1.0),
                link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
                link_masses=[1.0, m_distal],
                joint_frames=[identity_xform, child_anchor_back],
                link_inertias=[I, I],
            )
            model = builder.finalize(device=self.device)
            state = model.state()
            inverse_dynamics = model.inverse_dynamics()

            sweep = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
            for q in sweep:
                joint_q = state.joint_q.numpy()
                joint_q[0] = q
                state.joint_q.assign(joint_q)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                newton.eval_inverse_dynamics(
                    model=model,
                    state=state,
                    eval_type=newton.InverseDynamics.EvalType.GRAVITY_COMPENSATION_FORCE,
                    inverse_dynamics=inverse_dynamics,
                )
                tau = inverse_dynamics.gravity_compensation_force.numpy()

                expected = m_distal * g_mag * arm_length * np.cos(q)
                np.testing.assert_allclose(
                    -tau[0],
                    expected,
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=f"At q = {q}: expected {expected}, got -tau = {-tau[0]}",
                )


class TestCoriolisCompForce(TestInverseDynamicsBase):
    """Coriolis-compensation-force tests for the two-link pendulum harness."""

    def test_coriolis_zero_at_rest(self):
        """C(q, q_dot) must vanish when q_dot = 0."""
        builder = self._build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)
        state.joint_qd.zero_()

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(
            model,
            state,
            newton.InverseDynamics.EvalType.CORIOLIS_COMPENSATION_FORCE,
            inverse_dynamics,
        )

        tau = inverse_dynamics.coriolis_compensation_force.numpy()
        np.testing.assert_allclose(tau, np.zeros_like(tau), atol=1e-6)

    def test_coriolis_double_pendulum_matches_analytical(self):
        """C(q, q_dot)*q_dot for a planar 3D double pendulum matches PhysX's analytical reference values.

        Replicates the PhysX ``CoriolisAndCentrifugalCompensationForces``
        unit test in
        ``physics/physx/test/unittests/Articulation/src/ArticulationInverseDynamics.cpp``:
        a fixed-base double pendulum with two revolute joints both about
        world +Y, link length 1.0, and a 25 kg point mass at each link
        midpoint. At ``q = (0, pi/2)`` link 1 points along world +X and
        link 2 along world -Z. Because both joint axes lie along world Y,
        motion is planar in the X-Z plane and the Coriolis term collapses
        to a closed form that is independent of the link rotational
        inertias:

            c_1 = -m * L_1 * l_2c * sin(q2) * (2 * q_dot1 * q_dot2 + q_dot2^2)
            c_2 =  m * L_1 * l_2c * sin(q2) * q_dot1^2

        With ``m = 25``, ``L_1 = 1``, ``l_2c = 0.5`` the prefactor is 12.5,
        which yields the two PhysX-quoted reference cases below.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        pos_half = wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity())
        neg_half = wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity())
        y_axis = wp.vec3(0.0, 1.0, 0.0)

        builder = newton.ModelBuilder(gravity=-10.0, up_axis=newton.Axis.Z)

        b1 = builder.add_link(
            xform=identity_xform,
            mass=25.0,
            inertia=self.I_UNIT,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=y_axis,
            parent_xform=identity_xform,
            child_xform=neg_half,
        )
        b2 = builder.add_link(
            xform=identity_xform,
            mass=25.0,
            inertia=self.I_UNIT,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        j2 = builder.add_joint_revolute(
            parent=b1,
            child=b2,
            axis=y_axis,
            parent_xform=pos_half,
            child_xform=neg_half,
        )
        builder.add_articulation([j1, j2], label="double_pendulum")

        model = builder.finalize(device=self.device)
        inverse_dynamics = model.inverse_dynamics()

        # Both states share q = (0, pi/2); only q_dot varies. Expected values are
        # +C(q, q_dot) * q_dot per the PhysX convention.
        cases = [
            ((1.5, 0.0), (0.0, 28.125)),
            ((1.5, 1.5), (-84.375, 28.125)),
        ]

        for joint_qd_values, expected_physx in cases:
            with self.subTest(joint_qd=joint_qd_values):
                state = model.state()
                joint_q = state.joint_q.numpy()
                joint_q[:] = (0.0, np.pi / 2.0)
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = joint_qd_values
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                newton.eval_inverse_dynamics(
                    model,
                    state,
                    newton.InverseDynamics.EvalType.CORIOLIS_COMPENSATION_FORCE,
                    inverse_dynamics,
                )

                # Newton's coriolis_compensation_force has the same sign
                # convention as gravity_compensation_force (i.e. opposite
                # of "force the user would apply to compensate"), so
                # compare -tau against the PhysX values.
                measured = inverse_dynamics.coriolis_compensation_force.numpy()
                np.testing.assert_allclose(-measured, expected_physx, atol=1e-3, rtol=1e-5)

    def test_coriolis_radial_slider_matches_analytical(self):
        """C(q, q_dot)*q_dot for a rotating radial slider matches the closed-form values.

        The classic Coriolis textbook example: a fixed-base 2-DOF
        articulation where Link 0 is attached to the world by a revolute
        joint about world +Z and Link 1 (the slider) is attached to
        Link 0 by a prismatic joint along Link 0's local +X. Link 0 is
        given a small but non-zero mass so the dominant translational
        inertia comes from the slider (a 0.5 kg point mass on the
        rotating axis). Drives and friction are disabled explicitly so
        the only joint loads come from inertial coupling (the RNEA
        compensation pass already zeroes these internally, but we make
        the intent explicit at the model level).

        With ``q = (theta, r)`` and ``q_dot = (omega, v_r)`` the slider
        traces a circle of varying radius and the kinetic energy is
        ``0.5 * m * (v_r^2 + r^2 * omega^2)``. The Coriolis term reduces
        to:

            c_theta = 2 * m * r * omega * v_r   (Coriolis coupling)
            c_r     = -m * r * omega^2          (centrifugal pull)

        ``theta`` is irrelevant because the system is rotationally
        symmetric about world +Z. Link rotational inertia is irrelevant
        too: the angular velocity vector is purely along +Z, so only
        each link's ``I[2, 2]`` enters the kinetic energy, and that
        component is invariant under rotations about Z, so it adds a
        constant offset to ``M[0, 0]`` but contributes nothing to
        ``dM/dq``. The outer loop sweeps three qualitatively different
        link inertias (negligible, unit, and 100x unit) to confirm this.
        """
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        z_axis = wp.vec3(0.0, 0.0, 1.0)
        x_axis = wp.vec3(1.0, 0.0, 0.0)

        m_slider = 0.5
        m_base = 1e-6

        omega = 2.0
        v_r = 0.1
        r = 1.0
        theta = 0.7  # arbitrary; system is rotationally symmetric about +Z

        expected = (
            2.0 * m_slider * r * omega * v_r,
            -m_slider * r * omega * omega,
        )

        I_negligible = wp.mat33(1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6)
        for link_inertia in (I_negligible, *self.INERTIA_PASSES):
            with self.subTest(inertia=link_inertia):
                builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)

                base = builder.add_link(
                    xform=identity_xform,
                    mass=m_base,
                    inertia=link_inertia,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j_rot = builder.add_joint_revolute(
                    parent=-1,
                    child=base,
                    axis=z_axis,
                    parent_xform=identity_xform,
                    child_xform=identity_xform,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
                slider = builder.add_link(
                    xform=identity_xform,
                    mass=m_slider,
                    inertia=link_inertia,
                    com=wp.vec3(0.0, 0.0, 0.0),
                )
                j_slide = builder.add_joint_prismatic(
                    parent=base,
                    child=slider,
                    axis=x_axis,
                    parent_xform=identity_xform,
                    child_xform=identity_xform,
                    target_ke=0.0,
                    target_kd=0.0,
                    friction=0.0,
                )
                builder.add_articulation([j_rot, j_slide], label="radial_slider")

                model = builder.finalize(device=self.device)
                inverse_dynamics = model.inverse_dynamics()

                state = model.state()
                joint_q = state.joint_q.numpy()
                joint_q[:] = (theta, r)
                state.joint_q.assign(joint_q)
                joint_qd = state.joint_qd.numpy()
                joint_qd[:] = (omega, v_r)
                state.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                newton.eval_inverse_dynamics(
                    model,
                    state,
                    newton.InverseDynamics.EvalType.CORIOLIS_COMPENSATION_FORCE,
                    inverse_dynamics,
                )

                measured = inverse_dynamics.coriolis_compensation_force.numpy()
                np.testing.assert_allclose(-measured, expected, atol=1e-4, rtol=1e-5)


class TestMassMatrix(TestInverseDynamicsBase):
    """Mass-matrix tests for the two-link pendulum harness."""

    def test_mass_matrix_matches_eval_mass_matrix(self):
        """eval_inverse_dynamics(EvalType.MASS_MATRIX) must match newton.eval_mass_matrix element-wise."""
        builder = self._build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            link_coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
            link_masses=[1.0, 2.0],
            joint_frames=[
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            ],
            link_inertias=[self.I_UNIT, self.I_UNIT],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)

        H_reference = newton.eval_mass_matrix(model, state).numpy()

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(model, state, newton.InverseDynamics.EvalType.MASS_MATRIX, inverse_dynamics)

        np.testing.assert_allclose(inverse_dynamics.mass_matrix.numpy(), H_reference, rtol=1e-6, atol=1e-6)


class TestGravCompForceCPU(TestGravCompForce, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestGravCompForceCUDA(TestGravCompForce, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestMassMatrixCPU(TestMassMatrix, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestMassMatrixCUDA(TestMassMatrix, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestCoriolisCompForceCPU(TestCoriolisCompForce, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestCoriolisCompForceCUDA(TestCoriolisCompForce, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


class TestManipulatorEquationCPU(TestManipulatorEquation, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestManipulatorEquationCUDA(TestManipulatorEquation, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


if __name__ == "__main__":
    unittest.main()
