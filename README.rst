|Tests Status| |Coverage|

AeHMC provides MCMC sampling algorithms written in `Aesara <https://github.com/pymc-devs/aesara>`_.

Features
========
- Sample from an (unnormalized) probability distribution using Hamiltonian Monte
  Carlo and the No U-Turn Sampler.

Example
=======

.. code-block:: python

    import aesara
    from aesara import tensor as at
    from aesara.tensor.random.utils import RandomStream

    from aeppl import joint_logprob

    from aehmc import nuts

    # A simple normal distribution
    Y_rv = at.random.normal(0, 1)


    def logprob_fn(y):
        return joint_logprob({Y_rv: y})


    # Build the transition kernel
    srng = RandomStream(seed=0)
    kernel = nuts.new_kernel(srng, logprob_fn)

    # Compile a function that updates the chain
    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    step_size = at.as_tensor(1e-2)
    inverse_mass_matrix=at.as_tensor(1.0)
    (
        next_state,
        potential_energy,
        potential_energy_grad,
        acceptance_prob,
        num_doublings,
        is_turning,
        is_diverging,
    ), updates = kernel(*initial_state, step_size, inverse_mass_matrix)

    next_step_fn = aesara.function([y_vv], next_state, updates=updates)

    print(next_step_fn(0))
    # 0.14344008534533775


Installation
============

The latest release of AeHMC can be installed from PyPI using ``pip``:

::

    pip install aehmc

Or via conda-forge:

::

    conda install -c conda-forge aehmc


The current development branch of AeHMC can be installed from GitHub using ``pip``:

::

    pip install git+https://github.com/aesara-devs/aehmc



.. |Tests Status| image:: https://github.com/aesara-devs/aehmc/actions/workflows/test.yml/badge.svg?branch=main
  :target: https://github.com/aesara-devs/aehmc/actions/workflows/test.yml
.. |Coverage| image:: https://codecov.io/gh/aesara-devs/aehmc/branch/main/graph/badge.svg?token=L2i59LsFc0
  :target: https://codecov.io/gh/aesara-devs/aehmc
