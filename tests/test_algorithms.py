import aesara
import aesara.tensor as at
import pytest

from aehmc import algorithms


def test_dual_averaging():
    """Find the minimum of a simple function using Dual Averaging."""

    def fn(x):
        return (x - 1) ** 2

    init, update = algorithms.dual_averaging(gamma=0.5)

    def one_step(step, x, x_avg, gradient_avg):
        value = fn(x)
        gradient = aesara.grad(value, x)
        return update(gradient, step, x, x_avg, gradient_avg)

    x_init = at.as_tensor(0.0, dtype="floatX")
    step, x_avg, gradient_avg = init(x_init)

    states, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": step},
            {"initial": x_init},
            {"initial": x_avg},
            {"initial": gradient_avg},
        ],
        n_steps=100,
    )

    last_x_avg = states[1].eval()[-1]
    assert last_x_avg == pytest.approx(1.0, 1e-2)
