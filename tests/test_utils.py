from copy import copy
from types import MethodType

import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.exceptions import ShapeError

from aehmc.utils import RaveledParamsMap


def test_RaveledParamsMap():
    size = (3, 2)
    tau_rv = at.random.invgamma(0.5, 0.5, name="tau")
    tau_vv = tau_rv.clone()
    tau_vv.name = "t"

    beta_rv = at.random.normal(0, at.sqrt(tau_rv), size=size, name="beta")
    beta_vv = beta_rv.clone()
    beta_vv.name = "b"

    params_map = {beta_rv: beta_vv, tau_rv: tau_vv}

    rp_map = RaveledParamsMap(params_map.keys())

    assert repr(rp_map) == "RaveledParamsMap((beta, tau))"

    q = at.vector("q")

    exp_beta_part = np.exp(np.arange(np.prod(size)).reshape(size))
    exp_tau_part = 1.0
    exp_raveled_params = np.concatenate(
        [exp_beta_part.ravel(), np.atleast_1d(exp_tau_part)]
    )

    raveled_params_at = rp_map.ravel_params([beta_vv, tau_vv])
    raveled_params_val = raveled_params_at.eval(
        {beta_vv: exp_beta_part, tau_vv: exp_tau_part}
    )

    assert np.array_equal(raveled_params_val, exp_raveled_params)

    unraveled_params_at = rp_map.unravel_params(q)

    beta_part = unraveled_params_at[beta_rv]
    tau_part = unraveled_params_at[tau_rv]

    new_test_point = {q: exp_raveled_params}
    assert np.array_equal(beta_part.eval(new_test_point), exp_beta_part)
    assert np.array_equal(tau_part.eval(new_test_point), exp_tau_part)


def test_RaveledParamsMap_bad_infer_shape():
    bad_normal_op = copy(at.random.normal)

    def bad_infer_shape(self, *args, **kwargs):
        raise ShapeError()

    bad_normal_op.infer_shape = MethodType(bad_infer_shape, bad_normal_op)

    size = (3, 2)
    beta_rv = bad_normal_op(0, 1, size=size, name="beta")
    beta_vv = beta_rv.clone()
    beta_vv.name = "b"

    params_map = {beta_rv: beta_vv}

    with pytest.warns(Warning):
        RaveledParamsMap(params_map.keys())
