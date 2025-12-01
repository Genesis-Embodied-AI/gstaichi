$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

# for uv
# $env:Path = "$HOME\.local\bin;$env:Path"
$env:GSTAICHI_CMAKE_ARGS = "-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_BUILD_TESTS:BOOL=ON"
python build.py
