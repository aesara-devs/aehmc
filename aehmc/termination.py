from typing import Callable, Tuple

import aesara
import aesara.tensor as aet
import numpy as np
from aesara.scan.utils import until
from aesara.tensor.var import TensorVariable


def iterative_uturn(is_turning_fn: Callable):
    """U-Turn termination criterion to check reversiblity while expanding
    the trajectory in a multiplicative way.

    The code follows the implementation in Numpyro [0]_, which is equivalent to
    that in TFP [1]_.

    References
    ----------
    .. [0]: Phan, Du, Neeraj Pradhan, and Martin Jankowiak. "Composable effects
            for flexible and accelerated probabilistic programming in NumPyro." arXiv
            preprint arXiv:1912.11554 (2019).
    .. [1]: Lao, Junpeng, et al. "tfp. mcmc: Modern markov chain monte carlo
            tools built for modern hardware." arXiv preprint arXiv:2002.01184 (2020).

    """

    def new_state(
        state: TensorVariable, max_num_doublings: int
    ) -> Tuple[TensorVariable, TensorVariable, int, int]:
        num_dims = state.ndims
        return (
            aet.zeros(max_num_doublings, num_dims),
            aet.zeros(max_num_doublings, num_dims),
            0,
            0,
        )

    def update(
        state: Tuple, momentum_sum: TensorVariable, momentum: TensorVariable, step: int
    ):
        momentum_ckpt, momentum_sum_ckpt, *_ = state
        idx_min, idx_max = _find_storage_indices(step)
        update_momentum_cktp, update_momentum_sum_ckpt = aesara.ifelse(
            step % 2 == 0,
            (momentum, momentum_sum),
            (momentum_ckpt, momentum_sum_ckpt[idx_max]),
        )
        momentum_ckpt[idx_max] = update_momentum_cktp
        momentum_sum_ckpt[idx_max] = update_momentum_sum_ckpt

        return (momentum_ckpt, momentum_sum_ckpt, idx_min, idx_max)

    def _find_storage_indices(step):
        """Convert a trajectory length to the ids between which the relevant
        momenta and momentum sums will be stored respectively in
        `momentum_ckpt` and `momentum_sum_ckpt.

        """

        def count_subtrees(nc):
            nc[0] = nc[0] >> 1
            nc[1] = nc[1] + 1
            return nc, until(nc[0] & 1 == 0)

        ncs, _ = aesara.scan(
            count_subtrees,
            outputs_info=(aet.constant(step), aet.constant(0)),
            n_steps=step,
        )
        num_subtrees = ncs[-1][1]

        def find_idx_max(nc):
            nc[0] = nc[0] >> 1
            nc[1] = nc[1] + (nc[0] & 1)
            return nc, until(nc[1] <= 0)

        ncs, _ = aesara.scan(
            find_idx_max, (aet.constant(step >> 1), aet.constant(0)), n_steps=step
        )
        idx_max = ncs[-1][1]

        idx_min = idx_max - num_subtrees + 1

        return idx_min, idx_max

    def is_iterative_turning(
        state: Tuple, momentum_sum: TensorVariable, momentum: TensorVariable
    ):
        """Check if a U-Turn happened in the current trajectory.

        To do so we iterate over the recorded momenta and sums of momentum that
        are relevant. None if this is an odd node, otherwise the leftmost node of
        all subtrees for which the current node is the rightmost node. The system
        is such that all the relevant nodes are stored between `idx_min` and `idx_max`
        """
        momentum_ckpts, momentum_sum_ckpts, idx_min, idx_max = state

        def body_fn(i):
            subtree_momentum_sum = (
                momentum_sum - momentum_sum_ckpts[i] + momentum_ckpts[i]
            )
            is_turning = is_turning_fn(momentum_ckpts[i], momentum, subtree_momentum_sum)
            do_stop = aet.lt(i - 1, idx_min) or is_turning
            return (i - 1, is_turning), until(do_stop)

        val, _ = aesara.scan(body_fn, outputs_info=(idx_max, None), n_steps=idx_max + 2)
        # is_turning = turning_values[-1]

        return val

    return new_state, update, is_iterative_turning
