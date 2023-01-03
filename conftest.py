import os


def pytest_sessionstart(session):
    os.environ["AESARA_FLAGS"] = ",".join(
        [
            os.environ.setdefault("AESARA_FLAGS", ""),
            "floatX=float64,on_opt_error=raise,on_shape_error=raise,cast_policy=numpy+floatX",
        ]
    )
