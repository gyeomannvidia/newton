# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverNotifyFlags

class TestFixedTendon(unittest.TestCase):
    def run_test_fixed_tendon_limits(self):
        world_up_axis = 0
        g = 0
        dt = 0.001

        # Scene complexity
        nb_worlds = 2
        nb_articulations_per_world = 2

        # Description of joints that are referenced by a tendon
        nb_dynamic_links_per_articulation = 2
        fixed_tendon_gearing_multpliers= [1.0, 0.5]
        joint_types = [newton.JointType.PRISMATIC, newton.JointType.PRISMATIC]
        joint_motion_axes = [1, 1]

        # Same mass properties for all linkss
        body_mass = 1.0
        body_inertia =  wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        body_com = wp.vec3(0.0, 0.0, 0.0)

        # Create a single world with an articulation count of nb_articulations_per_world 
        # and a link count per articulation of nb_dynamic_links_per_articulation
        individual_world_builder = newton.ModelBuilder(gravity=g, up_axis=world_up_axis)
        for i in range(0, nb_articulations_per_world):            
    
            # Record all joint ids - needed for add_articulation()
            joint_ids = []
            # Record the joint ids to be referenced by the fixed tendon
            fixed_tendon_joint_ids = []
            fixed_tendon_gearings = []

            # Create the root link with a fixed joint to the world.
            # Note: SolverMujoco requires at least one shape per articulation so we'll add a shape here
            # even though we don't need any to test a tendon.
            root_body_index = individual_world_builder.add_link(mass=body_mass, I_m=body_inertia, armature=0.0, com=body_com)
            #individual_world_builder.add_shape_sphere(
            #    radius=1.0, body=root_body_index, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False))
            joint_id = individual_world_builder.add_joint_fixed(
                parent = -1,
                child = root_body_index,
                parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
            )
            joint_ids.append(joint_id)

            for j in range (0, nb_dynamic_links_per_articulation):

                # Create a child of the root with either a prismatic or revolute joint as inbound joint
                # Set joint friction to 0 to make predictions easier.
                link0_index = individual_world_builder.add_link(mass=body_mass, I_m=body_inertia, armature=0.0, com=body_com)
                individual_world_builder.add_shape_sphere(
                    radius=1.0, body=link0_index, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False))

                if joint_types[j] == newton.JointType.PRISMATIC:
                    joint_id = individual_world_builder.add_joint_prismatic(
                            axis = joint_motion_axes[j],
                            parent = root_body_index,
                            child = link0_index,
                            friction = 0)

                else:
                    joint_id = individual_world_builder.add_joint_revolute(
                            axis = joint_motion_axes[j],
                            parent = root_body_index,
                            child = link0_index,
                            friction = 0)

                joint_ids.append(joint_id)  
                fixed_tendon_joint_ids.append(joint_id)
                fixed_tendon_gearings.append((i+1)*fixed_tendon_gearing_multpliers[j])

            # All links and joints have been added to the articulation.
            # We can now create a tendon that couples the joints.
            individual_world_builder.add_fixed_tendon(joints=fixed_tendon_joint_ids, gearings=fixed_tendon_gearings, L_min=-1.0, L_max=1.0)

            # That's the articulation complete.
            individual_world_builder.add_articulation(joint_ids)

        # Create nb_worlds copies of the original world configuration
        main_builder = newton.ModelBuilder(gravity=g, up_axis=world_up_axis)
        for i in range(0, nb_worlds):
            main_builder.add_builder(individual_world_builder, world=i)

        # Create the MujocoSolver instance
        model = main_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        Solver = SolverMuJoCo(model, iterations=1, ls_iterations=1, disable_contacts=True, use_mujoco_cpu=False, integrator="euler")
    
    def test_fixed_tendon_limits(self):
        self.run_test_fixed_tendon_limits()

if __name__ == "__main__":
    unittest.main(verbosity=2)
