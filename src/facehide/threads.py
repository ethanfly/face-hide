from __future__ import annotations

import os

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_DEFAULT = 4
_MIN = 1
_MAX = 8


def clamped_facehide_threads() -> int:
    raw = os.environ.get("FACEHIDE_THREADS", str(_DEFAULT))
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        n = _DEFAULT
    return max(_MIN, min(_MAX, n))


def apply_env() -> int:
    n = clamped_facehide_threads()
    for key in _THREAD_KEYS:
        os.environ[key] = str(n)
    return n


def intra_op() -> int:
    return clamped_facehide_threads()


def apply(*, dml_active: bool = False) -> None:
    import cv2

    n = clamped_facehide_threads()
    cv2.setNumThreads(1 if dml_active else n)
