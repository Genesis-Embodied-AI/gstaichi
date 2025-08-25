# Adapted from https://github.com/duburcqa/jiminy/blob/dev/build_tools/wheel_addplat_macos.py
# which was written by Alexis Duburcq, under MIT license
# - wrapped name with NormalizedName, to satisfy typing checker
# - imported delocate_addplat.main as delocate_addplat to make it clear
#   that running `main` runs delocate_addplat
import sys
from typing import Tuple, FrozenSet

import packaging.utils
from packaging.utils import NormalizedName, BuildTag
from packaging.version import Version
from packaging.tags import Tag

parse_wheel_filename_orig = packaging.utils.parse_wheel_filename

def parse_wheel_filename(
    filename: str,
) -> Tuple[NormalizedName, Version, BuildTag, FrozenSet[Tag]]:
    name, version, build, tags = parse_wheel_filename_orig(filename)
    return NormalizedName(name.replace("-", "_")), version, build, tags

packaging.utils.parse_wheel_filename = parse_wheel_filename

from delocate.cmd.delocate_addplat import main as delocate_addplat

if __name__ == '__main__':
    sys.exit(delocate_addplat())
