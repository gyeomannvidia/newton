# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mockup of a full USD pipeline test.

``NewtonSimulationAPI`` does not exist yet. ``SimulationConfig`` and
``create_from_usd`` below stand in for the real implementation so the intended
behavior can be pinned down first.
"""

import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

import newton

try:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

SIMULATION_API = "NewtonSimulationAPI"
# All three APIs are applied to the same UsdPhysics.Scene prim.
SCENE_APIS = (SIMULATION_API, "NewtonCollisionPipelineAPI", "NewtonMuJoCoSceneAPI")
COLLISION_PIPELINE_API = SCENE_APIS[1]
# Maps the SimulationConfig.solver string to (solver class, API that configures it).
# The Kamino and Featherstone APIs and their create_from_usd functions are mocked up.
SOLVERS = {
    "mujoco": (newton.solvers.SolverMuJoCo, "NewtonMuJoCoSceneAPI"),
    "kamino": (newton.solvers.SolverKamino, "NewtonKaminoSceneAPI"),
    "featherstone": (newton.solvers.SolverFeatherstone, "NewtonFeatherstoneSceneAPI"),
}


def _has_api(prim, api: str) -> bool:
    """Return whether ``api`` is authored in ``apiSchemas``, even if the schema is not registered."""
    api_schemas = prim.GetMetadata("apiSchemas")
    return api_schemas is not None and api in api_schemas.GetAddedOrExplicitItems()


def _require_api(prim, api: str) -> None:
    if not _has_api(prim, api):
        raise ValueError(f"{prim.GetPath()}: {api} is not applied.")


@dataclass
class SimulationConfig:
    """Simulation settings authored by ``NewtonSimulationAPI`` on a physics scene."""

    num_substeps: int = 1
    collision_decimation: int = 1
    device: str = "cpu"
    use_cuda_graph: bool = False
    num_worlds: int = 1
    solver: str = "mujoco"
    """Solver to instantiate with ``create_from_usd``; one of the keys of :data:`SOLVERS`."""

    @classmethod
    def create_from_usd(cls, prim) -> "SimulationConfig":
        """Create a config from the ``newton:simulation:*`` attributes of ``prim``.

        Only authored values override the defaults.

        Raises:
            ValueError: If the API is not applied or an authored value is invalid.
        """
        _require_api(prim, SIMULATION_API)

        def authored(name: str) -> Any:
            attr = prim.GetAttribute(f"newton:simulation:{name}")
            return attr.Get() if attr and attr.HasAuthoredValue() else None

        kwargs = {
            "num_substeps": authored("numSubsteps"),
            "collision_decimation": authored("collisionDecimation"),
            "device": authored("device"),
            "use_cuda_graph": authored("useCudaGraph"),
            "num_worlds": authored("numWorlds"),
            "solver": authored("solver"),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        for name in ("num_substeps", "collision_decimation", "num_worlds"):
            if name in kwargs and kwargs[name] < 1:
                raise ValueError(f"{prim.GetPath()}: {name} must be >= 1, got {kwargs[name]!r}.")
        if "solver" in kwargs and kwargs["solver"] not in SOLVERS:
            raise ValueError(f"{prim.GetPath()}: solver must be one of {sorted(SOLVERS)}, got {kwargs['solver']!r}.")
        return cls(**kwargs)


def _authored_solver(prim) -> str:
    attr = prim.GetAttribute("newton:simulation:solver")
    return attr.Get() if attr and attr.HasAuthoredValue() else SimulationConfig.solver


def uses_mujoco_contacts(prim, solver: str) -> bool:
    """Return whether the solver computes its own contacts, so Newton's collision pipeline is not needed.

    ``newton:mujoco:useMujocoContacts`` belongs to ``NewtonMuJoCoSceneAPI``, so it only applies to
    the ``"mujoco"`` solver. Contacts are MuJoCo's own unless it is authored as false (the solver's
    default is true).
    """
    if solver != "mujoco":
        return False
    attr = prim.GetAttribute("newton:mujoco:useMujocoContacts")
    return bool(attr.Get()) if attr and attr.HasAuthoredValue() else True


def validate_scene_apis(prim) -> None:
    """Check that every API the scene needs is applied, before anything is parsed.

    Requires ``NewtonSimulationAPI`` and the API of the solver selected by
    ``newton:simulation:solver`` (``NewtonMuJoCoSceneAPI`` for ``"mujoco"``).
    ``NewtonCollisionPipelineAPI`` is required unless the MuJoCo solver uses its own contacts.

    Raises:
        ValueError: If a required API is not applied.
    """
    _require_api(prim, SIMULATION_API)
    solver = _authored_solver(prim)
    required = []
    if not uses_mujoco_contacts(prim, solver):
        required.append(COLLISION_PIPELINE_API)
    if solver in SOLVERS:  # an unknown solver is reported by SimulationConfig.create_from_usd
        required.append(SOLVERS[solver][1])
    for api in required:
        _require_api(prim, api)


def read_gravity(prim) -> tuple[float, float, float]:
    """Return the gravity vector [m/s^2] in stage axes authored on a ``UsdPhysics.Scene`` prim.

    An unauthored (zero) ``gravityDirection`` points down the stage up axis and an unauthored
    (negative) ``gravityMagnitude`` is 9.81. Gravity is not part of :class:`SimulationConfig`.
    """
    scene = UsdPhysics.Scene(prim)
    direction = np.array(scene.GetGravityDirectionAttr().Get(), dtype=float)
    magnitude = float(scene.GetGravityMagnitudeAttr().Get())
    length = np.linalg.norm(direction)
    if length > 0.0:
        direction /= length
    else:
        up_axis = UsdGeom.GetStageUpAxis(prim.GetStage())
        direction = np.array([0.0, 1.0, 0.0] if up_axis == UsdGeom.Tokens.y else [0.0, 0.0, 1.0])
        direction = -direction
    if magnitude < 0.0:
        magnitude = 9.81
    return tuple(float(x) for x in direction * magnitude)


def read_physics_dt(prim) -> float:
    """Return the step duration [s], 1 / ``newton:timeStepsPerSecond`` (default 1000 steps/s)."""
    attr = prim.GetAttribute("newton:timeStepsPerSecond")
    steps_per_second = attr.Get() if attr and attr.HasAuthoredValue() else None
    if steps_per_second is None or steps_per_second <= 0:
        steps_per_second = 1000
    return 1.0 / steps_per_second


def parse(prim):
    """Build ``(model, solver, collision_pipeline, simulation_config, physics_dt)`` from a ``UsdPhysics.Scene`` prim and its stage.

    See :func:`validate_scene_apis` for the APIs the scene must apply. ``collision_pipeline`` is
    ``None`` if the MuJoCo solver uses its own contacts. ``physics_dt`` is the step duration [s],
    read from ``newton:timeStepsPerSecond``.
    The scene's :class:`SimulationConfig` also selects the device and the number of replicated worlds, and gravity comes from the scene's ``UsdPhysics.Scene`` attributes.
    """
    validate_scene_apis(prim)
    simulation_config = SimulationConfig.create_from_usd(prim)
    solver_cls, _solver_api = SOLVERS[simulation_config.solver]

    stage = prim.GetStage()
    up_axis = UsdGeom.GetStageUpAxis(stage)
    scene_builder = newton.ModelBuilder(up_axis=up_axis)
    # Lets solver-specific attributes (e.g. mjc:option:* for MuJoCo) reach the model.
    solver_cls.register_custom_attributes(scene_builder)
    scene_builder.add_usd(stage)
    builder = newton.ModelBuilder(up_axis=up_axis, gravity=read_gravity(prim))
    builder.replicate(scene_builder, simulation_config.num_worlds)
    model = builder.finalize(device=simulation_config.device)

    solver = solver_cls.create_from_usd(prim, model)
    # With MuJoCo contacts the solver collides internally, so there is no collision pipeline.
    collision_pipeline = None
    if not uses_mujoco_contacts(prim, simulation_config.solver):
        collision_pipeline = newton.CollisionPipeline.create_from_usd(prim, model)
    return model, solver, collision_pipeline, simulation_config, read_physics_dt(prim)


def simulate(model, solver, collision_pipeline, simulation_config, physics_dt: float, num_frames: int = 1):
    """Run ``num_frames`` frames and return the final state.

    The scene's :class:`SimulationConfig` selects the number of steps per frame and how often
    collision detection reruns. If the MuJoCo solver uses its own contacts, no Newton contacts are
    computed. Each step lasts ``physics_dt`` seconds. If
    ``use_cuda_graph`` is set and the device is CUDA, one frame is captured as a CUDA graph and
    replayed for every frame; on a CPU device the flag is ignored.
    """
    # With MuJoCo contacts there is no collision pipeline and the solver collides internally.
    use_newton_contacts = collision_pipeline is not None
    contacts = collision_pipeline.contacts() if use_newton_contacts else None
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    if use_newton_contacts:
        collision_pipeline.collide(state_0, contacts)  # initial contacts for the first step

    def simulate_frame():
        s0, s1 = state_0, state_1
        for i in range(simulation_config.num_substeps):
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, physics_dt)
            s0, s1 = s1, s0
            if use_newton_contacts and (i + 1) % simulation_config.collision_decimation == 0:
                collision_pipeline.collide(s0, contacts)
        if s0 is not state_0:
            # An odd step count leaves the result in state_1; keep it in state_0 so a captured
            # graph can be replayed from the same buffers every frame.
            state_0.assign(s0)

    graph = None
    if simulation_config.use_cuda_graph and model.device.is_cuda:
        with wp.ScopedCapture(device=model.device) as capture:
            simulate_frame()
        graph = capture.graph
    for _ in range(num_frames):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            simulate_frame()
    return state_0


def parse_and_simulate(usd_path: str, num_frames: int = 1):
    """Parse the first ``UsdPhysics.Scene`` of a USD file, run it, and return ``(simulation_config, model, state)``."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Failed to open USD stage {usd_path!r}.")
    scene_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scene_prims:
        raise ValueError(f"{usd_path}: the stage has no UsdPhysics.Scene prim.")
    prim = scene_prims[0]
    model, solver, collision_pipeline, simulation_config, physics_dt = parse(prim)
    state = simulate(model, solver, collision_pipeline, simulation_config, physics_dt, num_frames)
    return simulation_config, model, state


def _make_scene(apis: tuple[str, ...] = SCENE_APIS, **attrs: tuple[Any, Any]):
    """Return ``(stage, prim)`` for a physics scene with all three Newton scene APIs applied.

    The caller must keep ``stage`` alive for as long as ``prim`` is used.

    ``apis`` selects which APIs are applied. Each other keyword maps a ``newton:simulation:<name>`` attribute to a ``(Sdf type, value)`` pair.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    prim = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
    for api in apis:
        prim.AddAppliedSchema(api)
    for name, (type_name, value) in attrs.items():
        prim.CreateAttribute(f"newton:simulation:{name}", type_name).Set(value)
    return stage, prim


@unittest.skipIf(Usd is None, "Requires usd-core")
class TestFullUsdMockup(unittest.TestCase):
    def test_defaults_when_nothing_authored(self):
        _stage, prim = _make_scene()
        simulation_config = SimulationConfig.create_from_usd(prim)
        self.assertEqual(simulation_config, SimulationConfig())

    def test_reads_all_authored_values(self):
        _stage, prim = _make_scene(
            numSubsteps=(Sdf.ValueTypeNames.Int, 4),
            collisionDecimation=(Sdf.ValueTypeNames.Int, 2),
            device=(Sdf.ValueTypeNames.String, "cuda:0"),
            useCudaGraph=(Sdf.ValueTypeNames.Bool, True),
            numWorlds=(Sdf.ValueTypeNames.Int, 16),
            solver=(Sdf.ValueTypeNames.String, "mujoco"),
        )
        simulation_config = SimulationConfig.create_from_usd(prim)
        self.assertEqual(simulation_config.num_substeps, 4)
        self.assertEqual(simulation_config.collision_decimation, 2)
        self.assertEqual(simulation_config.device, "cuda:0")
        self.assertTrue(simulation_config.use_cuda_graph)
        self.assertEqual(simulation_config.num_worlds, 16)
        self.assertEqual(simulation_config.solver, "mujoco")

    def test_partial_authoring_keeps_defaults(self):
        _stage, prim = _make_scene(numWorlds=(Sdf.ValueTypeNames.Int, 8))
        simulation_config = SimulationConfig.create_from_usd(prim)
        self.assertEqual(simulation_config.num_worlds, 8)
        self.assertEqual(simulation_config.num_substeps, 1)

    def test_gravity_from_scene_attributes(self):
        stage, prim = _make_scene()
        scene = UsdPhysics.Scene(prim)
        scene.CreateGravityDirectionAttr().Set((0.0, -2.0, 0.0))
        scene.CreateGravityMagnitudeAttr().Set(3.0)
        np.testing.assert_allclose(read_gravity(prim), (0.0, -3.0, 0.0))

    def test_solver_selection(self):
        for name in SOLVERS:
            with self.subTest(solver=name):
                _stage, prim = _make_scene(solver=(Sdf.ValueTypeNames.String, name))
                self.assertEqual(SimulationConfig.create_from_usd(prim).solver, name)

    def test_parse_requires_selected_solver_api(self):
        for name, (_cls, solver_api) in SOLVERS.items():
            with self.subTest(solver=name):
                apis = (SIMULATION_API, COLLISION_PIPELINE_API)
                _stage, prim = _make_scene(apis, solver=(Sdf.ValueTypeNames.String, name))
                with self.assertRaisesRegex(ValueError, f"{solver_api} is not applied"):
                    parse(prim)

    def test_uses_mujoco_contacts(self):
        _stage, prim = _make_scene()
        simulation_config = SimulationConfig.create_from_usd(prim)
        self.assertTrue(uses_mujoco_contacts(prim, simulation_config.solver))  # solver default
        prim.CreateAttribute("newton:mujoco:useMujocoContacts", Sdf.ValueTypeNames.Bool).Set(False)
        self.assertFalse(uses_mujoco_contacts(prim, simulation_config.solver))
        prim.GetAttribute("newton:mujoco:useMujocoContacts").Set(True)
        self.assertTrue(uses_mujoco_contacts(prim, simulation_config.solver))
        self.assertFalse(uses_mujoco_contacts(prim, "featherstone"))

    def test_invalid_solver_raises(self):
        _stage, prim = _make_scene(solver=(Sdf.ValueTypeNames.String, "nonexistent"))
        with self.assertRaisesRegex(ValueError, "solver must be one of"):
            SimulationConfig.create_from_usd(prim)

    def test_missing_api_raises(self):
        stage = Usd.Stage.CreateInMemory()
        prim = UsdPhysics.Scene.Define(stage, "/physicsScene").GetPrim()
        with self.assertRaisesRegex(ValueError, "is not applied"):
            SimulationConfig.create_from_usd(prim)

    def test_invalid_value_raises(self):
        _stage, prim = _make_scene(numSubsteps=(Sdf.ValueTypeNames.Int, 0))
        with self.assertRaisesRegex(ValueError, "num_substeps must be >= 1"):
            SimulationConfig.create_from_usd(prim)

    def test_parse_and_simulate_requires_apis(self):
        mujoco_api = SOLVERS["mujoco"][1]
        # (missing API, whether MuJoCo contacts are disabled so the collision pipeline is needed)
        cases = [(SIMULATION_API, False), (mujoco_api, False), (COLLISION_PIPELINE_API, True)]
        for missing, disable_mujoco_contacts in cases:
            with self.subTest(missing=missing):
                apis = tuple(api for api in SCENE_APIS if api != missing)
                stage, prim = _make_scene(apis)
                if disable_mujoco_contacts:
                    prim.CreateAttribute("newton:mujoco:useMujocoContacts", Sdf.ValueTypeNames.Bool).Set(False)
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "scene.usda")
                    stage.GetRootLayer().Export(path)
                    with self.assertRaisesRegex(ValueError, f"{missing} is not applied"):
                        parse_and_simulate(path)

    def test_collision_pipeline_api_optional_with_mujoco_contacts(self):
        apis = tuple(api for api in SCENE_APIS if api != COLLISION_PIPELINE_API)
        _stage, prim = _make_scene(apis)
        validate_scene_apis(prim)  # MuJoCo contacts are the default, so no error

    @unittest.skipUnless(
        hasattr(newton.CollisionPipeline, "create_from_usd")
        and hasattr(newton.solvers.SolverMuJoCo, "create_from_usd"),
        "Requires CollisionPipeline.create_from_usd and SolverMuJoCo.create_from_usd "
        "(NewtonCollPipelineUSD and NewtonSolverMuJoCoUSD branches)",
    )
    def test_parse_and_simulate(self):
        stage, _prim = _make_scene(
            numSubsteps=(Sdf.ValueTypeNames.Int, 2),
            collisionDecimation=(Sdf.ValueTypeNames.Int, 2),
            numWorlds=(Sdf.ValueTypeNames.Int, 3),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.usda")
            stage.GetRootLayer().Export(path)
            simulation_config, model, state = parse_and_simulate(path)
        self.assertEqual(simulation_config.num_worlds, 3)
        self.assertEqual(model.world_count, 3)
        self.assertIsNotNone(state)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
