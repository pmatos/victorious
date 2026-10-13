"""The CPUs available to threaded libraries."""

import os
from pathlib import Path

SMT_FLAG = Path("/sys/devices/system/cpu/smt/active")


def physical_cores(smt_flag: Path = SMT_FLAG) -> int:
    """The CPUs this process may use, halved when the kernel reports hyper-threading as on.

    Matrix-heavy inference gains nothing from a second hardware thread per core and loses to
    the contention, so it is sized by cores.
    """
    logical = os.process_cpu_count() or 1
    try:
        threaded = smt_flag.read_text().strip() == "1"
    except OSError:
        threaded = False
    return max(1, logical // 2) if threaded else logical
