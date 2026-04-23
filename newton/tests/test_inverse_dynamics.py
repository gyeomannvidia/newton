# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for InverseDynamics, eval_inverse_dynamics(), and the
gravity/Coriolis compensation-force helpers."""

from __future__ import annotations

import unittest

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
        raise ValueError(
            f"gravity must have at most one non-zero component (axis-aligned); got {components}."
        )
    if non_zero:
        axis_idx = non_zero[0]
        return components[axis_idx], (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[axis_idx]
    return 0.0, newton.Axis.Y


def _build_two_link_pendulum(
    gravity: wp.vec3,
    floating_base: bool,
    joint_type: str,
    joint_axis: wp.vec3,
    coms: list[wp.vec3],
    link_masses: list[float],
    joint_frames: list[wp.transform] 
) -> newton.ModelBuilder:
    gravity_scalar, up_axis = _gravity_vec_to_scalar_and_axis(gravity)
    builder = newton.ModelBuilder(gravity=gravity_scalar, up_axis=up_axis)

    # Inertial properties are supplied directly; no collision shapes are needed
    # since inverse dynamics only consumes mass, inertia, and CoM.
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    if joint_type == "revolute":
        add_dof_joint = builder.add_joint_revolute
    elif joint_type == "prismatic":
        add_dof_joint = builder.add_joint_prismatic
    else:
        raise ValueError(f"joint_type must be 'revolute' or 'prismatic', got {joint_type!r}.")

    identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    b1 = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=link_masses[0],
        inertia=inertia,
        com=coms[0],
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
        inertia=inertia,
        com=coms[1],
    )
    j2 = add_dof_joint(
        parent=b1,
        child=b2,
        axis=joint_axis,
        parent_xform=joint_frames[0],
        child_xform=joint_frames[1]
    )
    builder.add_articulation([j1, j2], label="pendulum")

    return builder
class TestInverseDynamicsBase:
    """Shared test body. Concrete subclasses set :attr:`device`."""

    device: wp.context.Device | None = None


    def _test_two_link_gravity_compensation_force(
        self,
        gravity_vec: wp.vec3,
        joint_type: str,
        joint_axis: wp.vec3,
        is_floating_base: list[list[bool]],
        centre_of_masses: list[list[list[wp.vec3]]],
        link_masses: list[list[list[float]]],
        joint_frames: list[list[list[wp.transform]]],
        expected_gravity_comp_forces: list[float],
    ):
        """G(q) is populated correctly for a multi-world, multi-articulation model.

        Args:
            gravity_vec: Axis-aligned gravity (world frame).
            joint_type: ``"revolute"`` or ``"prismatic"`` — shared across every articulation.
            joint_axis: Shared joint axis used for every articulation.
            is_floating_base: Per-articulation floating-vs-fixed root flag ``[w][a]``.
            centre_of_masses: Per-link CoM offsets ``[w][a][link]`` as ``wp.vec3``.
            link_masses: Per-link masses ``[w][a][link]``.
            joint_frames: Per-joint parent-side anchor transforms ``[w][a][joint]``.
                ``joint[0]`` is the root joint (free or fixed), ``joint[1]`` is
                the internal DOF joint.
            expected_gravity_comp_forces: Flat expected ``-G(q)`` in the order
                Newton reports them.
        """
        gravity_scalar, up_axis = _gravity_vec_to_scalar_and_axis(gravity_vec)

        # Derive shape constants from the structured inputs.
        num_worlds = len(is_floating_base)
        num_arts_per_world = len(is_floating_base[0])
        num_links_per_articulation = len(centre_of_masses[0][0])
        # _build_two_link_pendulum hard-codes a two-link articulation, so the
        # caller's coms layout must agree.
        self.assertEqual(num_links_per_articulation, 2)

        # Build the model from the structured per-world / per-articulation inputs.
        model_builder = newton.ModelBuilder(gravity=gravity_scalar, up_axis=up_axis)
        for i in range(0, num_worlds):
            world_builder = newton.ModelBuilder(gravity=gravity_scalar, up_axis=up_axis)
            for j in range(0, num_arts_per_world):
                articulation_builder = _build_two_link_pendulum(
                    gravity=gravity_vec,
                    joint_type=joint_type,
                    joint_axis=joint_axis,
                    floating_base=is_floating_base[i][j],
                    coms=centre_of_masses[i][j],
                    link_masses=link_masses[i][j],
                    joint_frames=joint_frames[i][j])
                world_builder.add_builder(articulation_builder)
            model_builder.add_world(world_builder)

        model = model_builder.finalize(device=self.device)
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        inverse_dynamics = model.inverse_dynamics()

        # Instantiate SolverMuJoCo so its custom-attribute registrations
        # (eq_solref / eq_solimp / actuator knobs, etc.) round-trip into the
        # finalized model, then advance one simulation step so the solver
        # builds its internal mjw_model / mjw_data.#
        solver = newton.solvers.SolverMuJoCo(model)
        state_next = model.state()
        control = model.control()
        solver.step(state, state_next, control, None, 1.0 / 60.0)
        mujoco_tau = solver.mj_data.qfrc_bias.copy()          
        print(mujoco_tau)

        newton.eval_inverse_dynamics(
            model=model,
            state=state,
            eval_type=newton.InverseDynamicsEvalType.EVAL_GRAVITY_COMPENSATION_FORCE,
            inverse_dynamics=inverse_dynamics,
        )

        measured_gravity_comp_force = inverse_dynamics.gravity_compensation_force.numpy()
        self.assertTrue(np.all(np.isfinite(measured_gravity_comp_force)))

        # Newton's gravity_compensation_force returns G(q); the force the user
        # would apply to hold the articulation static is -G(q), which is what
        # expected_gravity_comp_forces lists, so compare against -tau.
        self.assertEqual(measured_gravity_comp_force.shape, (len(expected_gravity_comp_forces),))
        np.testing.assert_allclose(
            -measured_gravity_comp_force, expected_gravity_comp_forces, atol=1e-5, rtol=1e-5
        )

    def test_two_link_prismatic_gravity_compensation_force_from_zero_gravity(self):      

        is_floating_base = [
            [False, True],  # World0, artic0 fixed, artic1 free
            [False, True]   # World1, artic0 fixed, artic1 free
        ]
        coms = [            
            [ 
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic1, link0/link1
            ],
            [ 
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic1, link0/link1
            ]
        ]       
        link_masses = [
                [[1.0, 2.0], [3.0, 4.0]], # World0,
                [[5.0, 6.0], [7.0, 8.0]]  # World1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform], # World0, artic0, root/internal joint
                [identity_xform, identity_xform], # World0, artic1, root/internal joint
            ],
            [
                [identity_xform, identity_xform], # World1, artic0, root/internal joint
                [identity_xform, identity_xform], # World1, artic1, root/internal joint
            ],
        ]

        gravity_vec = wp.vec3(0.0, 0.0, 0.0)
        joint_axis = wp.vec3(0.0, 1.0, 0.0)

        expected_gravity_comp_forces = [
            0.0,                                 # World 0, fixed root, 1 dof
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # World 0, floating root, 6+1 dofs
            0.0,                                 # World 1, fixed root, 1 dof
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 80.0,  # World 1, floating root, 6+1 dofs
        ]


    def test_two_link_prismatic_gravity_compensation_force_from_mass(self):

        is_floating_base = [
            [False, True],  # World0, artic0 fixed, artic1 free
            [False, True]   # World1, artic0 fixed, artic1 free
        ]
        coms = [            
            [ 
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic1, link0/link1
            ],
            [ 
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic1, link0/link1
            ]
        ]       
        link_masses = [
                [[1.0, 2.0], [3.0, 4.0]], # World0,
                [[5.0, 6.0], [7.0, 8.0]]  # World1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform], # World0, artic0, root/internal joint
                [identity_xform, identity_xform], # World0, artic1, root/internal joint
            ],
            [
                [identity_xform, identity_xform], # World1, artic0, root/internal joint
                [identity_xform, identity_xform], # World1, artic1, root/internal joint
            ],
        ]

        gravity_vec = wp.vec3(0.0, -10.0, 0.0)
        joint_axis = wp.vec3(0.0, 1.0, 0.0)

        expected_gravity_comp_forces = [
            20.0,                                # World 0, fixed root, 1 dof
            0.0, 70.0, 0.0, 0.0, 0.0, 0.0, 40,   # World 0, floating root, 6+1 dofs
            60.0,                                # World 1, fixed root, 1 dof
            0.0, 150.0, 0.0, 0.0, 0.0, 0.0, 80,  # World 1, floating root, 6+1 dofs
        ]

        self._test_two_link_gravity_compensation_force(
            gravity_vec, "prismatic", joint_axis, is_floating_base, coms, link_masses, joint_frames, expected_gravity_comp_forces
        )

    def test_two_link_prismatic_gravity_compensation_force_from_com(self):

        is_floating_base = [
            [False, True],  # World0, artic0 fixed, artic1 free
            [False, True]   # World1, artic0 fixed, artic1 free
        ]
        coms = [            
            [ 
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic0, link0/link1
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic1, link0/link1
            ],
            [ 
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic0, link0/link1
                [wp.vec3(0.5, 0.0, 0.0), wp.vec3(0.5, 0.0, 0.0)], # World1, artic1, link0/link1
            ]
        ]       
        link_masses = [
                [[1.0, 2.0], [3.0, 4.0]], # World0,
                [[5.0, 6.0], [7.0, 8.0]]  # World1
        ]

        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, identity_xform], # World0, artic0, root/internal joint
                [identity_xform, identity_xform], # World0, artic1, root/internal joint
            ],
            [
                [identity_xform, identity_xform], # World1, artic0, root/internal joint
                [identity_xform, identity_xform], # World1, artic1, root/internal joint
            ],
        ]

        gravity_vec = wp.vec3(0.0, -10.0, 0.0)
        joint_axis = wp.vec3(0.0, 1.0, 0.0)

        expected_gravity_comp_forces = [
            20.0,                                # World 0, fixed root, 1 dof
            0.0, 70.0, 0.0, 0.0, 0.0, 15, 40,    # World 0, floating root, 6+1 dofs
            60.0,                                # World 1, fixed root, 1 dof
            0.0, 150.0, 0.0, 0.0, 0.0, 75.0, 80, # World 1, floating root, 6+1 dofs
        ]

        self._test_two_link_gravity_compensation_force(
            gravity_vec, "prismatic", joint_axis, is_floating_base, coms, link_masses, joint_frames, expected_gravity_comp_forces
        )


    def test_two_link_revolute_gravity_compensation_force(self):

        is_floating_base = [
            [False, True],  # World0, artic0 fixed, artic1 free
            [False, True]   # World1, artic0 fixed, artic1 free
        ]
        coms = [
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World0, artic1, link0/link1
            ],
            [
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic0, link0/link1
                [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], # World1, artic1, link0/link1
            ]
        ]
        link_masses = [
                [[1.0, 2.0], [1.0, 2.0]], # World0,
                [[1.0, 2.0], [1.0, 2.0]]  # World1
        ]

        # Root link sits at the origin (its root joint has identity xforms
        # inside _build_two_link_pendulum). The revolute joint anchor is also
        # at the origin (parent_xform = identity in the root's body frame),
        # and the child body's origin is placed at (4, 0, 0) by putting the
        # joint anchor at (-4, 0, 0) in the child's body frame. Newton resolves
        # child_body_pose = parent_xform · joint_q · inv(child_xform), which
        # at q = 0 gives identity · identity · (4, 0, 0, I) = (4, 0, 0, I).
        identity_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
        child_anchor_back = wp.transform(wp.vec3(-4.0, 0.0, 0.0), wp.quat_identity())
        joint_frames = [
            [
                [identity_xform, child_anchor_back], # World0, artic0, internal joint parent/child xforms
                [identity_xform, child_anchor_back], # World0, artic1, internal joint parent/child xforms
            ],
            [
                [identity_xform, child_anchor_back], # World1, artic0, internal joint parent/child xforms
                [identity_xform, child_anchor_back], # World1, artic1, internal joint parent/child xforms
            ],
        ]

        gravity_vec = wp.vec3(0.0, -10.0, 0.0)
        joint_axis = wp.vec3(0.0, 0.0, 1.0)

        expected_gravity_comp_forces = [
            80.0,                                    # World 0, fixed root, 1 dof
            0.0, 30.0, 0.0, 0.0, 0.0, 80.0, 80.0,    # World 0, floating root, 6+1 dofs
            80.0,                                    # World 1, fixed root, 1 dof
            0.0, 30.0, 0.0, 0.0, 0.0, 80.0, 80.0,    # World 1, floating root, 6+1 dofs
        ]

        self._test_two_link_gravity_compensation_force(
            gravity_vec, "revolute", joint_axis, is_floating_base, coms, link_masses, joint_frames, expected_gravity_comp_forces
        )


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



    def test_mass_matrix_matches_eval_mass_matrix(self):
        """eval_inverse_dynamics(EVAL_MASS_MATRIX) must match newton.eval_mass_matrix element-wise."""
        builder = _build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)

        H_reference = newton.eval_mass_matrix(model, state).numpy()

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(
            model, state, newton.InverseDynamicsEvalType.EVAL_MASS_MATRIX, inverse_dynamics
        )

        np.testing.assert_allclose(
            inverse_dynamics.mass_matrix.numpy(), H_reference, rtol=1e-6, atol=1e-6
        )

    def test_coriolis_zero_at_rest(self):
        """C(q, q_dot) must vanish when q_dot = 0."""
        builder = _build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)
        state.joint_qd.zero_()

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(
            model,
            state,
            newton.InverseDynamicsEvalType.EVAL_CORIOLIS_COMPENSATION_FORCE,
            inverse_dynamics,
        )

        tau = inverse_dynamics.coriolis_compensation_force.numpy()
        np.testing.assert_allclose(tau, np.zeros_like(tau), atol=1e-6)

    def test_gravity_zero_without_gravity(self):
        """G(q) must vanish when the model has zero gravity."""
        builder = _build_two_link_pendulum(
            gravity=wp.vec3(0.0, 0.0, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(
            model,
            state,
            newton.InverseDynamicsEvalType.EVAL_GRAVITY_COMPENSATION_FORCE,
            inverse_dynamics,
        )

        tau = inverse_dynamics.gravity_compensation_force.numpy()
        np.testing.assert_allclose(tau, np.zeros_like(tau), atol=1e-6)

    def test_gravity_nonzero_under_gravity(self):
        """G(q) is generically non-zero for a non-trivial pose under gravity."""
        builder = _build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=False,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(
            model,
            state,
            newton.InverseDynamicsEvalType.EVAL_GRAVITY_COMPENSATION_FORCE,
            inverse_dynamics,
        )

        tau = inverse_dynamics.gravity_compensation_force.numpy()
        self.assertGreater(float(np.linalg.norm(tau)), 1e-6)

    def test_eval_all_populates_every_buffer(self):
        """EVAL_ALL must write the mass matrix and both compensation forces in one call.

        Uses a floating base so the articulation has multi-DOF coupling; a
        fixed root with a single revolute DOF has identically-zero Coriolis
        and would trivially defeat that assertion.
        """
        builder = _build_two_link_pendulum(
            gravity=wp.vec3(0.0, -9.81, 0.0),
            floating_base=True,
            joint_type="revolute",
            joint_axis=wp.vec3(0.0, 0.0, 1.0),
            coms=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)],
        )
        model = builder.finalize(device=self.device)
        state = self._pose_pendulum(model)
        joint_qd = state.joint_qd.numpy()
        # Populate linear, angular, and internal DOFs so Coriolis has real
        # coupling: pure linear base motion doesn't couple through C(q, q_dot).
        joint_qd[:] = np.linspace(0.1, 0.7, joint_qd.shape[0])
        state.joint_qd.assign(joint_qd)

        inverse_dynamics = model.inverse_dynamics()
        newton.eval_inverse_dynamics(
            model, state, newton.InverseDynamicsEvalType.EVAL_ALL, inverse_dynamics
        )

        H = inverse_dynamics.mass_matrix.numpy()
        g = inverse_dynamics.gravity_compensation_force.numpy()
        c = inverse_dynamics.coriolis_compensation_force.numpy()

        self.assertTrue(np.all(np.isfinite(H)))
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertGreater(float(np.linalg.norm(H)), 1e-6)
        self.assertGreater(float(np.linalg.norm(g)), 1e-6)
        self.assertGreater(float(np.linalg.norm(c)), 1e-6)


class TestInverseDynamicsCPU(TestInverseDynamicsBase, unittest.TestCase):
    device = wp.get_device("cpu")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA not available")
class TestInverseDynamicsCUDA(TestInverseDynamicsBase, unittest.TestCase):
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else None


if __name__ == "__main__":
    unittest.main()
