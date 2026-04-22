# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..core.types import Devicelike
from .enums import InverseDynamicsEvalType

if TYPE_CHECKING:
    from .model import Model
    from .state import State


class InverseDynamics:
    """Inverse dynamics quantities for a single articulated rigid-body system."""

    def __init__(
        self,
        num_dofs: int,
        floating_base: bool = False,
        device: Devicelike | None = None,
    ):
        """Allocate the joint-space mass matrix for one articulation.

        The matrix is stored row-major in a flat buffer of length
        ``size * size``, where ``size = num_dofs + 6`` for a floating-base
        articulation and ``size = num_dofs`` for a fixed-base articulation.

        Args:
            num_dofs: Number of internal joint DOFs.
            floating_base: Whether the root link is a floating base.
            device: Warp device on which the buffer is allocated.
        """
        size = num_dofs + 6 if floating_base else num_dofs
        self.mass_matrix: wp.array[wp.float32] = wp.zeros(size * size, dtype=wp.float32, device=device)


def eval_inverse_dynamics(
    model: Model,
    state: State,
    eval_type: InverseDynamicsEvalType,
    inverse_dynamics: InverseDynamics,
) -> None:
    """Compute inverse dynamics quantities for an articulation.

    Depending on the flags in ``eval_type``, populates one or more of:
    the joint-space mass matrix M(q), the gravity compensation force G(q),
    and the Coriolis compensation force C(q, q_dot) into ``inverse_dynamics``.

    Args:
        model: Model providing articulation topology and inertial parameters.
        state: State providing the current generalized coordinates and velocities.
        eval_type: Bitmask selecting which quantities to compute.
        inverse_dynamics: Output container whose buffers are written in place.
    """
