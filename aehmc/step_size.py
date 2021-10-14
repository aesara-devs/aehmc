from typing import Callable, Tuple

import aesara
import aesara.tensor as at
from aesara.scan.utils import until
from aesara.tensor.var import TensorVariable


def heuristic_adaptation(
    kernel: Callable,
    reference_state: Tuple,
    initial_step_size: TensorVariable,
    target_acceptance_rate=0.65,
    max_num_iterations=100,
):
    def update(step_size, direction, previous_direction):
        step_size = (2.0 ** direction) * step_size
        *_, p_accept = kernel(*reference_state, step_size)
        new_direction = at.where(
            at.lt(target_acceptance_rate, p_accept), at.constant(1), at.constant(-1)
        )
        return (step_size.astype("floatX"), new_direction, direction), until(
            at.neq(direction, previous_direction)
        )

    (step_sizes, _, _), _ = aesara.scan(
        fn=update,
        outputs_info=[
            {"initial": initial_step_size},
            {"initial": at.constant(0)},
            {"initial": at.constant(0)},
        ],
        n_steps=max_num_iterations,
    )

    return step_sizes[-1]
