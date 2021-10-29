"""Test dynamic termination criteria."""
import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from aehmc.metrics import gaussian_metric
from aehmc.termination import iterative_uturn


@pytest.mark.parametrize(
    "checkpoint_idxs, expected_turning",
    [((3, 3), True), ((0, 0), False), ((0, 1), True), ((1, 3), True)],
)
def test_iterative_turning_termination(checkpoint_idxs, expected_turning):
    inverse_mass_matrix = aet.as_tensor(np.ones(1))
    _, _, is_turning = gaussian_metric(inverse_mass_matrix)
    _, _, is_iterative_turning = iterative_uturn(is_turning)

    momentum = aet.scalar("momentum")
    momentum_sum = aet.scalar("momentum_sum")

    idx_min, idx_max = checkpoint_idxs
    idx_min = aet.as_tensor(idx_min)
    idx_max = aet.as_tensor(idx_max)
    momentum_ckpts = aet.as_tensor(np.array([1.0, 2.0, 3.0, -2.0]))
    momentum_sum_ckpts = aet.as_tensor(np.array([2.0, 4.0, 4.0, -1.0]))
    ckpt_state = (momentum_ckpts, momentum_sum_ckpts, idx_min, idx_max)

    _, _, is_iterative_turning_fn = iterative_uturn(is_turning)
    is_iterative_turning = is_iterative_turning_fn(ckpt_state, momentum_sum, momentum)
    fn = aesara.function(
        (momentum, momentum_sum), is_iterative_turning, on_unused_input="ignore"
    )

    actual_turning = fn(1.0, 3.0)

    assert actual_turning.ndim == 0
    assert expected_turning == actual_turning


@pytest.mark.parametrize(
    "num_dims",
    [1, 3],
)
def test_termination_update(num_dims):
    inverse_mass_matrix = aet.as_tensor(np.ones(1))
    _, _, is_turning = gaussian_metric(inverse_mass_matrix)
    new_state, update, _ = iterative_uturn(is_turning)

    position = aet.as_tensor(np.ones(num_dims))
    momentum = aet.as_tensor(np.ones(num_dims))
    momentum_sum = aet.as_tensor(np.ones(num_dims))

    num_doublings = aet.as_tensor(4)
    termination_state = new_state(position, num_doublings)

    step = aet.scalar("step", dtype="int32")
    updated = update(termination_state, momentum_sum, momentum, step)
    update_fn = aesara.function((step,), updated, on_unused_input="ignore")

    # Make sure this works for a single step
    result_odd = update_fn(1)

    # When the number of steps is odd there should be no update
    result_odd = update_fn(5)
    assert_array_equal(result_odd[0], np.zeros((4, num_dims)))
    assert_array_equal(result_odd[1], np.zeros((4, num_dims)))
