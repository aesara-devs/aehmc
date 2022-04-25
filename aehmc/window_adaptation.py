from typing import Callable, List, Tuple

import aesara
import aesara.tensor as at
from aesara import config
from aesara.tensor.shape import shape_tuple
from aesara.tensor.var import TensorVariable

from aehmc.mass_matrix import covariance_adaptation
from aehmc.step_size import dual_averaging_adaptation


def run(
    kernel_factory,
    initial_state,
    num_steps=1000,
    *,
    is_mass_matrix_full=False,
    initial_step_size=at.as_tensor(1.0, dtype=config.floatX),
    target_acceptance_rate=0.80
):
    init, update, final = window_adaptation(
        kernel_factory, is_mass_matrix_full, initial_step_size, target_acceptance_rate
    )

    def one_step(
        stage,  # schedule
        is_middle_window_end,
        q,  # chain state
        potential_energy,
        potential_energy_grad,
        step,  # Dual Averagin adaptation state
        log_step_size,
        log_step_size_avg,
        gradient_avg,
        mu,
        mean,  # Mass matrix adaptation state
        m2,
        sample_size,
    ):
        chain_state = (q, potential_energy, potential_energy_grad)
        warmup_state = (
            (step, log_step_size, log_step_size_avg, gradient_avg, mu),
            (mean, m2, sample_size),
        )

        chain_state, warmup_state, updates = update(
            stage, is_middle_window_end, chain_state, warmup_state
        )

        return (*chain_state, *warmup_state[0], *warmup_state[1]), updates

    schedule = build_schedule(num_steps)
    stage = at.as_tensor([s[0] for s in schedule])
    is_middle_window_end = at.as_tensor([s[1] for s in schedule])

    da_state, mm_state = init(initial_state)
    state, updates = aesara.scan(
        fn=one_step,
        outputs_info=(*initial_state, *da_state, *mm_state),
        sequences=(stage, is_middle_window_end),
    )

    last_chain_state = (state[0][-1], state[1][-1], state[2][-1])
    last_warmup_state = (
        (state[3][-1], state[4][-1], state[5][-1], state[6][-1], state[7][-1]),
        (state[8][-1], state[9][-1], state[10][-1]),
    )

    step_size, inverse_mass_matrix = final(last_warmup_state)

    return (last_chain_state, (step_size, inverse_mass_matrix)), updates


def window_adaptation(
    kernel_factory: Callable[[TensorVariable], Callable],
    is_mass_matrix_full: bool = False,
    initial_step_size: TensorVariable = at.as_tensor(1.0, dtype=config.floatX),
    target_acceptance_rate: TensorVariable = 0.80,
):
    mm_init, mm_update, mm_final = covariance_adaptation(is_mass_matrix_full)
    da_init, da_update = dual_averaging_adaptation(target_acceptance_rate)

    def init(initial_chain_state: Tuple):
        if initial_chain_state[0].ndim == 0:
            num_dims = 0
        else:
            num_dims = shape_tuple(initial_chain_state[0])[0]
        inverse_mass_matrix, mm_state = mm_init(num_dims)

        kernel = kernel_factory(inverse_mass_matrix)
        step_size = heuristic_adaptation(
            kernel, initial_chain_state, initial_step_size, target_acceptance_rate
        )
        step, logstepsize_avg, gradient_avg, mu = da_init(step_size)

        return (step, at.log(step_size), logstepsize_avg, gradient_avg, mu), mm_state

    def fast_update(p_accept, da_state, mm_state):
        da_state = da_update(p_accept, *da_state)
        return (da_state, mm_state)

    def slow_update(position, p_accept, da_state, mm_state):
        da_state = da_update(p_accept, *da_state)
        mm_state = mm_update(position, mm_state)
        return (da_state, mm_state)

    def slow_final(warmup_state):
        """We recompute the inverse mass matrix and re-initialize the dual averaging scheme at the end of each 'slow window'."""
        da_state, mm_state = warmup_state
        mm_final(mm_state)
        step, logstepsize_avg, gradient_avg, mu = da_init(at.exp(da_state[2]))
        return ((step, da_state[2], logstepsize_avg, gradient_avg, mu), mm_state)

    def update(
        stage: int, is_middle_window_end: bool, chain_state: Tuple, warmup_state: Tuple
    ):
        da_state, mm_state = warmup_state
        step_size = at.exp(da_state[2])
        inverse_mass_matrix = mm_final(mm_state)

        kernel = kernel_factory(inverse_mass_matrix)
        (*chain_state, p_accept, _, _, _), updates = kernel(*chain_state, step_size)

        warmup_state = where_warmup_state(
            at.eq(stage, 0),
            fast_update(p_accept, da_state, mm_state),
            slow_update(chain_state[0], p_accept, da_state, mm_state),
        )
        warmup_state = where_warmup_state(
            is_middle_window_end, slow_final(warmup_state), warmup_state
        )

        return chain_state, warmup_state, updates

    def where_warmup_state(do_pick_left, left_warmup_state, right_warmup_state):
        (
            left_step,
            left_logstepsize,
            left_logstepsize_avg,
            left_gradient_avg,
            left_mu,
        ) = left_warmup_state[0]
        (
            right_step,
            right_logstepsize,
            right_logstepsize_avg,
            right_gradient_avg,
            right_mu,
        ) = right_warmup_state[0]

        step = at.where(do_pick_left, left_step, right_step)
        logstepsize = at.where(do_pick_left, left_logstepsize, right_logstepsize)
        logstepsize_avg = at.where(
            do_pick_left, left_logstepsize_avg, right_logstepsize_avg
        )
        gradient_avg = at.where(do_pick_left, left_gradient_avg, right_gradient_avg)
        mu = at.where(do_pick_left, left_mu, right_mu)

        right_mean, right_m2, right_sample_size = right_warmup_state[1]
        left_mean, left_m2, left_sample_size = left_warmup_state[1]
        mean = at.where(do_pick_left, left_mean, right_mean)
        m2 = at.where(do_pick_left, left_m2, right_m2)
        sample_size = at.where(do_pick_left, left_sample_size, right_sample_size)

        return (step, logstepsize, logstepsize_avg, gradient_avg, mu), (
            mean,
            m2,
            sample_size,
        )

    def final(warmup_state: Tuple) -> Tuple[TensorVariable, TensorVariable]:
        da_state, mm_state = warmup_state
        step_size = at.exp(da_state[3])
        inverse_mass_matrix = mm_final(mm_state)
        return step_size, inverse_mass_matrix

    return init, update, final


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
        schedule += [(0, False)] * (num_steps - 1)
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
