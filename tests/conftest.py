"""Run distributed checks in isolated JAX CPU runtimes."""

import os
import re
import subprocess
import sys

import pytest


@pytest.fixture
def run_on_cpu_devices():
    def run(script, device_count, *arguments):
        environment = os.environ.copy()
        flags = re.sub(
            r"(?:^|\s)--xla_force_host_platform_device_count=\d+", "",
            environment.get("XLA_FLAGS", ""),
        ).strip()
        environment["XLA_FLAGS"] = (
            f"{flags} --xla_force_host_platform_device_count={device_count}".strip()
        )
        environment["JAX_PLATFORMS"] = "cpu"
        environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        result = subprocess.run(
            [sys.executable, "-c", script, *arguments], capture_output=True,
            text=True, env=environment, timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if result.stdout:
            print(result.stdout, end="")

    return run
