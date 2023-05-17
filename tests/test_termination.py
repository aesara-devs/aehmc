"""Test dynamic termination criteria."""
import aesara
import aesara.tensor as at
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aehmc.metrics import gaussian_metric
from aehmc.termination import TerminationState, _find_storage_indices, iterative_uturn


@pytest.mark.parametrize(
    "checkpoint_idxs, momentum, momentum_sum, inverse_mass_matrix, expected_turning",
    [
        ((3, 3), at.as_tensor(1.0), at.as_tensor(3.0), at.as_tensor(1.0), True),
        ((3, 2), at.as_tensor(1.0), at.as_tensor(3.0), at.as_tensor(1.0), False),
        ((0, 0), at.as_tensor(1.0), at.as_tensor(3.0), at.as_tensor(1.0), False),
        ((0, 1), at.as_tensor(1.0), at.as_tensor(3.0), at.as_tensor(1.0), True),
        ((1, 3), at.as_tensor(1.0), at.as_tensor(3.0), at.as_tensor(1.0), True),
        ((1, 3), at.as_tensor([1.0]), at.as_tensor([3.0]), at.ones(1), True),
    ],
)
def test_iterative_turning_termination(
    checkpoint_idxs, momentum, momentum_sum, inverse_mass_matrix, expected_turning
):
    _, _, is_turning = gaussian_metric(inverse_mass_matrix)
    _, _, is_iterative_turning = iterative_uturn(is_turning)

    idx_min, idx_max = checkpoint_idxs
    idx_min = at.as_tensor(idx_min)
    idx_max = at.as_tensor(idx_max)
    momentum_ckpts = at.as_tensor(np.array([1.0, 2.0, 3.0, -2.0]))
    momentum_sum_ckpts = at.as_tensor(np.array([2.0, 4.0, 4.0, -1.0]))
    ckpt_state = TerminationState(
        momentum_checkpoints=momentum_ckpts,
        momentum_sum_checkpoints=momentum_sum_ckpts,
        min_index=idx_min,
        max_index=idx_max,
    )

    _, _, is_iterative_turning_fn = iterative_uturn(is_turning)
    is_iterative_turning = is_iterative_turning_fn(ckpt_state, momentum_sum, momentum)
    fn = aesara.function((), is_iterative_turning, on_unused_input="ignore")

    actual_turning = fn()

    assert actual_turning.ndim == 0
    assert expected_turning == actual_turning


@pytest.mark.parametrize(
    "step, expected_idx",
    [(0, (1, 0)), (6, (3, 2)), (7, (0, 2)), (13, (2, 2)), (15, (0, 3))],
)
def test_leaf_idx_to_ckpt_idx(step, expected_idx):
    step_tt = at.scalar("step", dtype=np.int64)
    idx_tt = _find_storage_indices(step_tt)
    fn = aesara.function((step_tt,), (*idx_tt,))

    idx_vv = fn(step)
    assert idx_vv[0].item() == expected_idx[0]
    assert idx_vv[1].item() == expected_idx[1]


@pytest.mark.parametrize(
    "num_dims",
    [1, 3],
)
def test_termination_update(num_dims):
    inverse_mass_matrix = at.as_tensor(np.ones(1))
    _, _, is_turning = gaussian_metric(inverse_mass_matrix)
    new_state, update, _ = iterative_uturn(is_turning)

    position = at.as_tensor(np.ones(num_dims))
    momentum = at.as_tensor(np.ones(num_dims))
    momentum_sum = at.as_tensor(np.ones(num_dims))

    num_doublings = at.as_tensor(4)
    termination_state = new_state(position, num_doublings)

    step = at.scalar("step", dtype=np.int64)
    updated = update(termination_state, momentum_sum, momentum, step)
    update_fn = aesara.function((step,), updated, on_unused_input="ignore")

    # Make sure this works for a single step
    result_odd = update_fn(1)

    # When the number of steps is odd there should be no update
    result_odd = update_fn(5)
    assert_array_equal(result_odd[0], np.zeros((4, num_dims)))
    assert_array_equal(result_odd[1], np.zeros((4, num_dims)))
