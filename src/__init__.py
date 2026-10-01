"""macOS only: PyTorch and FAISS each ship an OpenMP runtime; on Mac they clash and crash.
Loading torch first and limiting threads avoids it. Has no effect on Kaggle (Linux)."""
import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch  # noqa: F401  (must load before faiss)
    except ImportError:
        pass
