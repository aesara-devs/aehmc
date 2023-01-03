import aesara.tensor as at
import numpy as np
import pytest
from aesara.graph.basic import Apply
from aesara.tensor.exceptions import ShapeError
from aesara.tensor.random.basic import NormalRV

from aehmc.utils import RaveledParamsMap


def test_RaveledParamsMap():
    tau_rv = at.random.invgamma(0.5, 0.5, name="tau")
    tau_vv = tau_rv.clone()
    tau_vv.name = "t"

    beta_size = (3, 2)
    beta_rv = at.random.normal(0, at.sqrt(tau_rv), size=beta_size, name="beta")
    beta_vv = beta_rv.clone()
    beta_vv.name = "b"

    kappa_size = (20,)
    kappa_rv = at.random.normal(0, 1, size=kappa_size, name="kappa")
    kappa_vv = kappa_rv.clone()
    kappa_vv.name = "k"

    params_map = {beta_rv: beta_vv, tau_rv: tau_vv, kappa_rv: kappa_vv}

    rp_map = RaveledParamsMap(params_map.keys())

    assert repr(rp_map) == "RaveledParamsMap((beta, tau, kappa))"

    q = at.vector("q")

    exp_beta_part = np.exp(np.arange(np.prod(beta_size)).reshape(beta_size))
    exp_tau_part = 1.0
    exp_kappa_part = np.exp(np.arange(np.prod(kappa_size)).reshape(kappa_size))
    exp_raveled_params = np.concatenate(
        [exp_beta_part.ravel(), np.atleast_1d(exp_tau_part), exp_kappa_part.ravel()]
    )

    raveled_params_at = rp_map.ravel_params([beta_vv, tau_vv, kappa_vv])
    raveled_params_val = raveled_params_at.eval(
        {beta_vv: exp_beta_part, tau_vv: exp_tau_part, kappa_vv: exp_kappa_part}
    )

    assert np.array_equal(raveled_params_val, exp_raveled_params)

    unraveled_params_at = rp_map.unravel_params(q)

    beta_part = unraveled_params_at[beta_rv]
    tau_part = unraveled_params_at[tau_rv]
    kappa_part = unraveled_params_at[kappa_rv]

    new_test_point = {q: exp_raveled_params}
    assert np.array_equal(beta_part.eval(new_test_point), exp_beta_part)
    assert np.array_equal(tau_part.eval(new_test_point), exp_tau_part)
    assert np.array_equal(kappa_part.eval(new_test_point), exp_kappa_part)


def test_RaveledParamsMap_dtype():
    tau_rv = at.random.normal(0, 1, name="tau")
    tau_vv = tau_rv.clone()
    tau_vv.name = "t"

    lambda_rv = at.random.binomial(10, 0.5, name="lmbda")
    lambda_vv = lambda_rv.clone()
    lambda_vv.name = "l"

    params_map = {tau_rv: tau_vv, lambda_rv: lambda_vv}
    rp_map = RaveledParamsMap(params_map.keys())

    q = rp_map.ravel_params((tau_vv, lambda_vv))
    unraveled_params = rp_map.unravel_params(q)

    tau_part = unraveled_params[tau_rv]
    lambda_part = unraveled_params[lambda_rv]

    assert tau_part.dtype == tau_rv.dtype
    assert lambda_part.dtype == lambda_rv.dtype


def test_RaveledParamsMap_bad_infer_shape():
    class BadNormalRV(NormalRV):
        def make_node(self, *args, **kwargs):
            res = super().make_node(*args, **kwargs)
            # Drop static `Type`-level shape information
            rv_out = res.outputs[1]
            outputs = [
                res.outputs[0].clone(),
                at.tensor(dtype=rv_out.type.dtype, shape=(None,) * rv_out.type.ndim),
            ]
            return Apply(
                self,
                res.inputs,
                outputs,
            )

        def infer_shape(self, *args, **kwargs):
            raise ShapeError()

    bad_normal_op = BadNormalRV()

    size = (3, 2)
    beta_rv = bad_normal_op(0, 1, size=size, name="beta")
    beta_vv = beta_rv.clone()
    beta_vv.name = "b"

    params_map = {beta_rv: beta_vv}

    with pytest.warns(Warning):
        RaveledParamsMap(params_map.keys())
