$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

$env:TAICHI_CMAKE_ARGS="-DTI_WITH_VULKAN:BOOL=ON -DTI_WITH_CUDA:BOOL=ON -DTI_WITH_OPENGL:BOOL=OFF"
python build.py
