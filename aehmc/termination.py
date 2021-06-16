from typing import Callable, Tuple

import aesara
import aesara.tensor as aet
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
        position: TensorVariable, max_num_doublings: int
    ) -> Tuple[TensorVariable, TensorVariable, int, int]:
        num_dims = aet.atleast_1d(position).shape[0]
        return (
            aet.zeros((max_num_doublings, num_dims)),
            aet.zeros((max_num_doublings, num_dims)),
            aet.constant(0, dtype="int32"),
            aet.constant(0, dtype="int32"),
        )

    def update(
        state: Tuple,
        momentum_sum: TensorVariable,
        momentum: TensorVariable,
        step: TensorVariable,
    ):
        momentum_ckpt, momentum_sum_ckpt, *_ = state
        idx_min, idx_max = _find_storage_indices(step)

        momentum_ckpt = aet.where(
            aet.eq(step % 2, 0),
            aet.set_subtensor(momentum_ckpt[idx_max], momentum),
            momentum_ckpt,
        )
        momentum_sum_ckpt = aet.where(
            aet.eq(step % 2, 0),
            aet.set_subtensor(momentum_sum_ckpt[idx_max], momentum_sum),
            momentum_sum_ckpt,
        )

        return (momentum_ckpt, momentum_sum_ckpt, idx_min, idx_max)

    def _find_storage_indices(step):
        """Convert a trajectory length to the ids between which the relevant
        momenta and momentum sums will be stored respectively in
        `momentum_ckpt` and `momentum_sum_ckpt.

        """

        def count_subtrees(nc0, nc1):
            do_stop = aet.eq(nc0 & 1, 0)
            new_nc0 = nc0 // 2
            new_nc1 = nc1 + 1
            return (new_nc0, new_nc1), until(do_stop)

        (_, nc1), _ = aesara.scan(
            count_subtrees,
            outputs_info=(step, -1),
            n_steps=step + 1,
        )
        num_subtrees = nc1[-1]

        def find_idx_max(nc0, nc1):
            do_stop = aet.eq(nc0, 0)
            new_nc0 = nc0 // 2
            new_nc1 = nc1 + (nc0 & 1)
            return (new_nc0, new_nc1), until(do_stop)

        init = step // 2
        init_nc1 = aet.constant(0).astype("int32")
        (nc0, nc1), _ = aesara.scan(
            find_idx_max, outputs_info=(init, init_nc1), n_steps=step + 1
        )
        idx_max = nc1[-1]

        idx_min = idx_max - num_subtrees + 1

        return idx_min, idx_max

    def is_iterative_turning(
        state: Tuple, momentum_sum: TensorVariable, momentum: TensorVariable
    ):
        """Check if a U-Turn happened in the current trajectory.

        To do so we iterate over the recorded momenta and sums of momentum that
        are relevant. None if the current node is an odd node, otherwise the
        leftmost node of all subtrees for which the current node is the
        rightmost node. The momentum and sum of momentum corresponding to the
        nodes for which we need to check the U-Turn criterion are stored
        between `idx_min` and `idx_max` in `momentum_ckpts` and
        `momentum_sum_ckpts` respectively.

        """
        momentum_ckpts, momentum_sum_ckpts, idx_min, idx_max = state

        def body_fn(i):
            subtree_momentum_sum = (
                momentum_sum - momentum_sum_ckpts[i] + momentum_ckpts[i]
            )
            is_turning = is_turning_fn(
                momentum_ckpts[i], momentum, subtree_momentum_sum
            )
            reached_max_iteration = aet.lt(i - 1, idx_min)
            do_stop = aet.any(is_turning | reached_max_iteration)
            return (i - 1, is_turning), until(do_stop)

        val, _ = aesara.scan(body_fn, outputs_info=(idx_max, None), n_steps=idx_max + 2)
        is_turning = val[1][-1]

        return is_turning

    return new_state, update, is_iterative_turning
