import os
import subprocess
import sys
import tempfile

from tests import test_utils


@test_utils.test()
def test_pyi_stubs():
    test_code = """
import taichi._lib.core.taichi_python
reveal_type(taichi._lib.core.taichi_python)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "tmp_mypy_test.py")
        with open(test_file, "w") as f:
            f.write(test_code)

        res = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pyright",
                test_file,
            ]
        ).decode("utf-8")
        assert "unknown" not in res.lower()
