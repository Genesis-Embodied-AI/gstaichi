import os
import sys

import pytest

import gstaichi as ti

from tests import test_utils

if sys.platform != "linux" or os.uname()[4] not in ["arm64", "aarch64"]:
    pytest.skip("This module is only for linux on arm64, which doesn't support assert", allow_module_level=True)


@test_utils.test()
def test_assert_ignored():
    @ti.kernel
    def k1() -> None:
        assert False
        assert True

    k1()
