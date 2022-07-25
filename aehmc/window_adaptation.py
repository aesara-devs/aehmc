from typing import List, Tuple

import aesara
import aesara.tensor as at
from aesara import config
from aesara.ifelse import ifelse
from aesara.tensor.shape import shape_tuple
from aesara.tensor.var import TensorVariable

from aehmc.mass_matrix import covariance_adaptation
from aehmc.step_size import dual_averaging_adaptation


def run(
    kernel,
    initial_state,
    num_steps=1000,
    *,
    is_mass_matrix_full=False,
    initial_step_size=at.as_tensor(1.0, dtype=config.floatX),
    target_acceptance_rate=0.80
):

    init_adapt, update_adapt = window_adaptation(
        num_steps, is_mass_matrix_full, initial_step_size, target_acceptance_rate
    )

    def one_step(
        warmup_step,
        q,  # chain state
        potential_energy,
        potential_energy_grad,
        step,  # dual averaging adaptation state
        log_step_size,
        log_step_size_avg,
        gradient_avg,
        mu,
        mean,  # mass matrix adaptation state
        m2,
        sample_size,
        step_size,  # parameters
        inverse_mass_matrix,
    ):
        chain_state = (q, potential_energy, potential_energy_grad)
        warmup_state = (
            (step, log_step_size, log_step_size_avg, gradient_avg, mu),
            (mean, m2, sample_size),
        )
        parameters = (step_size, inverse_mass_matrix)

        # Advance the chain by one step
        chain_state, inner_updates = kernel(*chain_state, *parameters)

        # Update the warmup state and parameters
        warmup_state, parameters = update_adapt(
            warmup_step, warmup_state, parameters, chain_state
        )

        return (
            chain_state[0],  # q
            chain_state[1],  # potential_energy
            chain_state[2],  # potential_energy_grad
            *warmup_state[0],
            *warmup_state[1],
            *parameters,
        ), inner_updates

    (da_state, mm_state), parameters = init_adapt(initial_state)

    warmup_steps = at.arange(0, num_steps)
    state, updates = aesara.scan(
        fn=one_step,
        outputs_info=(*initial_state, *da_state, *mm_state, *parameters),
        sequences=(warmup_steps,),
        name="window_adaptation",
    )

    last_chain_state = (state[0][-1], state[1][-1], state[2][-1])
    step_size = state[-2][-1]
    inverse_mass_matrix = state[-1][-1]

    return last_chain_state, (step_size, inverse_mass_matrix), updates


def window_adaptation(
    num_steps: int,
    is_mass_matrix_full: bool = False,
    initial_step_size: TensorVariable = at.as_tensor(1.0, dtype=config.floatX),
    target_acceptance_rate: TensorVariable = 0.80,
):
    mm_init, mm_update, mm_final = covariance_adaptation(is_mass_matrix_full)
    da_init, da_update = dual_averaging_adaptation(target_acceptance_rate)
    schedule = build_schedule(num_steps)

    schedule_stage = at.as_tensor([s[0] for s in schedule])
    schedule_middle_window = at.as_tensor([s[1] for s in schedule])

    def init(initial_chain_state: Tuple):
        if initial_chain_state[0].ndim == 0:
            num_dims = 0
        else:
            num_dims = shape_tuple(initial_chain_state[0])[0]
        inverse_mass_matrix, mm_state = mm_init(num_dims)

        da_state = da_init(initial_step_size)
        step_size = at.exp(da_state[1])

        warmup_state = (da_state, mm_state)
        parameters = (step_size, inverse_mass_matrix)
        return warmup_state, parameters

    def fast_update(p_accept, warmup_state, parameters):
        da_state, mm_state = warmup_state
        _, inverse_mass_matrix = parameters

        new_da_state = da_update(p_accept, *da_state)
        step_size = at.exp(new_da_state[1])

        return (new_da_state, mm_state), (step_size, inverse_mass_matrix)

    def slow_update(position, p_accept, warmup_state, parameters):
        da_state, mm_state = warmup_state
        _, inverse_mass_matrix = parameters

        new_da_state = da_update(p_accept, *da_state)
        new_mm_state = mm_update(position, mm_state)
        step_size = at.exp(new_da_state[1])

        return (new_da_state, new_mm_state), (step_size, inverse_mass_matrix)

    def slow_final(warmup_state):
        """We recompute the inverse mass matrix and re-initialize the dual averaging scheme at the end of each 'slow window'."""
        da_state, mm_state = warmup_state

        inverse_mass_matrix = mm_final(mm_state)

        if inverse_mass_matrix.ndim == 0:
            num_dims = 0
        else:
            num_dims = shape_tuple(inverse_mass_matrix)[0]
        _, new_mm_state = mm_init(num_dims)

        step_size = at.exp(da_state[1])
        new_da_state = da_init(step_size)

        warmup_state = (new_da_state, new_mm_state)
        parameters = (step_size, inverse_mass_matrix)
        return warmup_state, parameters

    def final(
        warmup_state: Tuple, parameters: Tuple
    ) -> Tuple[TensorVariable, TensorVariable]:
        da_state, _ = warmup_state
        _, inverse_mass_matrix = parameters
        step_size = at.exp(da_state[2])  # return stepsize_avg at the end
        return step_size, inverse_mass_matrix

    def update(step: int, warmup_state: Tuple, parameters: Tuple, chain_state: Tuple):
        position, _, _, p_accept, *_ = chain_state

        stage = schedule_stage[step]
        warmup_state, parameters = where_warmup_state(
            at.eq(stage, 0),
            fast_update(p_accept, warmup_state, parameters),
            slow_update(position, p_accept, warmup_state, parameters),
        )

        is_middle_window_end = schedule_middle_window[step]
        warmup_state, parameters = where_warmup_state(
            is_middle_window_end, slow_final(warmup_state), (warmup_state, parameters)
        )

        is_last_step = at.eq(step, num_steps - 1)
        parameters = ifelse(is_last_step, final(warmup_state, parameters), parameters)

        return warmup_state, parameters

    def where_warmup_state(do_pick_left, left_warmup_state, right_warmup_state):
        (left_da_state, left_mm_state), left_params = left_warmup_state
        (right_da_state, right_mm_state), right_params = right_warmup_state

        da_state = ifelse(do_pick_left, left_da_state, right_da_state)
        mm_state = ifelse(do_pick_left, left_mm_state, right_mm_state)
        params = ifelse(do_pick_left, left_params, right_params)

        return (da_state, mm_state), params

    return init, update


def build_schedule(
    num_steps: int,
    initial_buffer_size: int = 75,
    final_buffer_size: int = 50,
    first_window_size: int = 25,
) -> List[Tuple[int, bool]]:
    """Return the schedule for Stan's warmup.

    The schedule below is intended to be as close as possible to Stan's _[1].
    The warmup period is split into three stages:
    1. An initial fast interval to reach the typical set. Only the step size is
    adapted in this window.
    2. "Slow" parameters that require global information (typically covariance)
    are estimated in a series of expanding intervals with no memory; the step
    size is re-initialized at the end of each window. Each window is twice the
    size of the preceding window.
    3. A final fast interval during which the step size is adapted using the
    computed mass matrix.
    Schematically:

    ```
    +---------+---+------+------------+------------------------+------+
    |  fast   | s | slow |   slow     |        slow            | fast |
    +---------+---+------+------------+------------------------+------+
    ```

    The distinction slow/fast comes from the speed at which the algorithms
    converge to a stable value; in the common case, estimation of covariance
    requires more steps than dual averaging to give an accurate value. See _[1]
    for a more detailed explanation.

    Fast intervals are given the label 0 and slow intervals the label 1.

    Note
    ----
    It feels awkward to return a boolean that indicates whether the current
    step is the last step of a middle window, but not for other windows. This
    should probably be changed to "is_window_end" and we should manage the
    distinction upstream.

    Parameters
    ----------
    num_steps: int
        The number of warmup steps to perform.
    initial_buffer: int
        The width of the initial fast adaptation interval.
    first_window_size: int
        The width of the first slow adaptation interval.
    final_buffer_size: int
        The width of the final fast adaptation interval.

    Returns
    -------
    A list of tuples (window_label, is_middle_window_end).

    References
    ----------
    .. [1]: Stan Reference Manual v2.22 Section 15.2 "HMC Algorithm"

    """
    schedule = []

    # Give up on mass matrix adaptation when the number of warmup steps is too small.
    if num_steps < 20:
        schedule += [(0, False)] * num_steps
    else:
        # When the number of warmup steps is smaller that the sum of the provided (or default)
        # window sizes we need to resize the different windows.
        if initial_buffer_size + first_window_size + final_buffer_size > num_steps:
            initial_buffer_size = int(0.15 * num_steps)
            final_buffer_size = int(0.1 * num_steps)
            first_window_size = num_steps - initial_buffer_size - final_buffer_size

        # First stage: adaptation of fast parameters
        schedule += [(0, False)] * (initial_buffer_size - 1)
        schedule.append((0, False))

        # Second stage: adaptation of slow parameters in successive windows
        # doubling in size.
        final_buffer_start = num_steps - final_buffer_size

        next_window_size = first_window_size
        next_window_start = initial_buffer_size
        while next_window_start < final_buffer_start:
            current_start, current_size = next_window_start, next_window_size
            if 3 * current_size <= final_buffer_start - current_start:
                next_window_size = 2 * current_size
            else:
                current_size = final_buffer_start - current_start
            next_window_start = current_start + current_size
            schedule += [(1, False)] * (next_window_start - 1 - current_start)
            schedule.append((1, True))

        # Last stage: adaptation of fast parameters
        schedule += [(0, False)] * (num_steps - 1 - final_buffer_start)
        schedule.append((0, False))

    return schedule
