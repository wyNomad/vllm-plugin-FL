#!/usr/bin/env python3
"""Run personal diagnostic probes on ub39, then stop only their container."""

import subprocess
from pathlib import Path


ROOT = Path("/workspace/wyNomad/results/diagnostics_16k_20261008_v2")
CONTAINER = "wynomad-minicpm-s2-gpu0"


def main():
    try:
        with (ROOT / "runner.log").open("w") as logfile:
            result = subprocess.run(
                [
                    "docker", "exec", CONTAINER, "python3",
                    "/workspace/benchmark_results/diagnostics_16k_20261008_v2/run_probes.py",
                ],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                check=False,
            )
        (ROOT / "runner.exit").write_text(str(result.returncode) + "\n")
    finally:
        with (ROOT / "container_stop.log").open("w") as logfile:
            subprocess.run(
                ["docker", "stop", CONTAINER],
                stdout=logfile,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=60,
            )


if __name__ == "__main__":
    main()
