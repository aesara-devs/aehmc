<div align="center">

# Aehmc

[![Pypi][pypi-badge]][pypi]
[![Gitter][gitter-badge]][gitter]
[![Discord][discord-badge]][discord]
[![Twitter][twitter-badge]][twitter]

AeHMC provides implementations for the HMC and NUTS samplers in [Aesara](https://github.com/aesara-devs/aesara).

[Features](#features) •
[Get Started](#get-started) •
[Install](#install) •
[Get help](#get-help) •
[Contribute](#contribute)

</div>

## Get started

``` python
import aesara
from aesara import tensor as at
from aesara.tensor.random.utils import RandomStream

from aeppl import joint_logprob

from aehmc import nuts

# A simple normal distribution
Y_rv = at.random.normal(0, 1)


def logprob_fn(y):
    return joint_logprob(realized={Y_rv: y})[0]


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
```

## Install

The latest release of AeHMC can be installed from PyPI using ``pip``:

``` bash
pip install aehmc
```

Or via conda-forge:

``` bash
conda install -c conda-forge aehmc
```

The current development branch of AeHMC can be installed from GitHub using ``pip``:

``` bash
pip install git+https://github.com/aesara-devs/aehmc
```

## Get help

Report bugs by opening an [issue][issues]. If you have a question regarding the usage of AeHMC, start a [discussion][discussions]. For real-time feedback or more general chat about AeHMC use our [Discord server][discord] or [Gitter room][gitter].

## Contribute

AeHMC welcomes contributions. A good place to start contributing is by looking at the [issues][issues].

If you want to implement a new feature, open a [discussion][discussions] or come chat with us on [Discord][discord] or [Gitter][gitter].

[contributors]: https://github.com/aesara-devs/aehmc/graphs/contributors
[contributors-badge]: https://img.shields.io/github/contributors/aesara-devs/aehmc?style=flat-square&logo=github&logoColor=white&color=ECEFF4
[discussions]: https://github.com/aesara-devs/aehmc/discussions
[downloads-badge]: https://img.shields.io/pypi/dm/aehmc?style=flat-square&logo=pypi&logoColor=white&color=8FBCBB
[discord]: https://discord.gg/h3sjmPYuGJ
[discord-badge]: https://img.shields.io/discord/1072170173785723041?color=81A1C1&logo=discord&logoColor=white&style=flat-square
[gitter]: https://gitter.im/aesara-devs/aehmc
[gitter-badge]: https://img.shields.io/gitter/room/aesara-devs/aehmc?color=81A1C1&logo=matrix&logoColor=white&style=flat-square
[issues]: https://github.com/aesara-devs/aehmc/issues
[releases]: https://github.com/aesara-devs/aehmc/releases
[twitter]: https://twitter.com/AesaraDevs
[twitter-badge]: https://img.shields.io/twitter/follow/AesaraDevs?style=social
[pypi]: https://pypi.org/project/aehmc/
[pypi-badge]: https://img.shields.io/pypi/v/aehmc?color=ECEFF4&logo=python&logoColor=white&style=flat-square
