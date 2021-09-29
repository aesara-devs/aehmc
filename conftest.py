import os


def pytest_sessionstart(session):
    os.environ["AESARA_FLAGS"] = ",".join(
        [
            os.environ.setdefault("AESARA_FLAGS", ""),
            "floatX=float64,warn__ignore_bug_before=all",
        ]
    )
