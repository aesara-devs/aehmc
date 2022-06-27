from typing import Callable, Tuple

import aesara
import aesara.tensor as at
import numpy as np
from aesara import config as config
from aesara.ifelse import ifelse
from aesara.scan.utils import until
from aesara.tensor.var import TensorVariable


def iterative_uturn(is_turning_fn: Callable) -> Tuple[Callable, Callable, Callable]:
    """U-Turn termination criterion to check reversiblity while expanding
    the trajectory.

    The code follows the implementation in Numpyro [0]_, which is equivalent to
    that in TFP [1]_.

    Parameter
    ---------
    is_turning_fn:
        A function which, given the new momentum and the sum of the momenta
        along the trajectory returns a boolean that indicates whether the
        trajectory is turning on itself. Depends on the metric.

    References
    ----------
    .. [0]: Phan, Du, Neeraj Pradhan, and Martin Jankowiak. "Composable effects
            for flexible and accelerated probabilistic programming in NumPyro." arXiv
            preprint arXiv:1912.11554 (2019).
    .. [1]: Lao, Junpeng, et al. "tfp. mcmc: Modern markov chain monte carlo
            tools built for modern hardware." arXiv preprint arXiv:2002.01184 (2020).

    """

    def new_state(
        position: TensorVariable, max_num_doublings: TensorVariable
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """Initialize the termination state

        Parameters
        ----------
        position
            Example chain position. Used to infer the shape of the arrays that
            store relevant momentam and momentum sums.
        max_num_doublings
            Maximum number of doublings allowed in the multiplicative
            expansion. Determines the maximum number of momenta and momentum
            sums to store.

        Returns
        -------
        A tuple that represents a new state for the termination criterion.

        """
        if position.ndim == 0:
            return (
                at.zeros(max_num_doublings, dtype=config.floatX),
                at.zeros(max_num_doublings, dtype=config.floatX),
                at.constant(0, dtype=np.int64),
                at.constant(0, dtype=np.int64),
            )
        else:
            num_dims = position.shape[0]
            return (
                at.zeros((max_num_doublings, num_dims), dtype=config.floatX),
                at.zeros((max_num_doublings, num_dims), dtype=config.floatX),
                at.constant(0, dtype=np.int64),
                at.constant(0, dtype=np.int64),
            )

    def update(
        state: Tuple,
        momentum_sum: TensorVariable,
        momentum: TensorVariable,
        step: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """Update the termination state.

        Parameters
        ----------
        state
            The current termination state
        momentum_sum
            The sum of all momenta along the trajectory
        momentum
            The current momentum on the trajectory
        step
            Current step in the trajectory integration (starting at 0)

        Return
        ------
        A tuple that represents the updated termination state.

        """
        momentum_ckpt, momentum_sum_ckpt, *_ = state
        idx_min, idx_max = ifelse(
            at.eq(step, 0), (state[2], state[3]), _find_storage_indices(step)
        )

        momentum_ckpt = at.where(
            at.eq(step % 2, 0),
            at.set_subtensor(momentum_ckpt[idx_max], momentum),
            momentum_ckpt,
        )
        momentum_sum_ckpt = at.where(
            at.eq(step % 2, 0),
            at.set_subtensor(momentum_sum_ckpt[idx_max], momentum_sum),
            momentum_sum_ckpt,
        )

        return (momentum_ckpt, momentum_sum_ckpt, idx_min, idx_max)

    def is_iterative_turning(
        state: Tuple, momentum_sum: TensorVariable, momentum: TensorVariable
    ) -> bool:
        """Check if any sub-trajectory is making a U-turn.

        If we visualize the trajectory as a balanced binary tree, the
        subtrajectories for which we need to check the U-turn criterion are the
        ones for which the current node is the rightmost node. The
        corresponding momenta and sums of momentum corresponding to the nodes
        for which we need to check the U-Turn criterion are stored between
        `idx_min` and `idx_max` in `momentum_ckpts` and `momentum_sum_ckpts`
        respectively.

        Parameters
        ----------
        state
            The current termination state
        momentum_sum
            The sum of all momenta along the trajectory
        momentum
            The current momentum on the trajectory
        step
            Current step in the trajectory integration (starting at 0)


        Return
        ------
        True if any sub-trajectory makes a U-turn, False otherwise.

        """
        momentum_ckpts, momentum_sum_ckpts, idx_min, idx_max = state

        def body_fn(i):
            subtree_momentum_sum = (
                momentum_sum - momentum_sum_ckpts[i] + momentum_ckpts[i]
            )
            is_turning = is_turning_fn(
                momentum_ckpts[i], momentum, subtree_momentum_sum
            )
            reached_max_iteration = at.lt(i - 1, idx_min)
            do_stop = at.any(is_turning | reached_max_iteration)
            return (i - 1, is_turning), until(do_stop)

        val, _ = aesara.scan(body_fn, outputs_info=(idx_max, None), n_steps=idx_max + 2)

        is_turning = val[1][-1]
        is_turning = at.where(
            at.lt(idx_max, idx_min), at.as_tensor(0, dtype="bool"), is_turning
        )

        return is_turning

    return new_state, update, is_iterative_turning


def _find_storage_indices(step: TensorVariable) -> Tuple[int, int]:
    """Find the indices between which the momenta and sums are stored.

    Parameter
    ---------
    step
        The current step in the trajectory integration.

    Return
    ------
    The min and max indices between which the values relevant to check the
    U-turn condition for the current step are stored.

    """

    def count_subtrees(nc0, nc1):
        do_stop = at.eq(nc0 & 1, 0)
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
        do_stop = at.eq(nc0, 0)
        new_nc0 = nc0 // 2
        new_nc1 = nc1 + (nc0 & 1)
        return (new_nc0, new_nc1), until(do_stop)

    init = at.as_tensor(step // 2, dtype=np.int64)
    init_nc1 = at.constant(0, dtype=np.int64)
    (nc0, nc1), _ = aesara.scan(
        find_idx_max, outputs_info=(init, init_nc1), n_steps=step + 1
    )
    idx_max = nc1[-1]

    idx_min = idx_max - num_subtrees + 1

    return idx_min, idx_max
