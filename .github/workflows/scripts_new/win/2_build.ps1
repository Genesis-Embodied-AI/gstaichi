$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

$env:TAICHI_CMAKE_ARGS = "-DTI_WITH_VULKAN:BOOL=ON -DTI_BUILD_TESTS:BOOL=ON"

python build.py
