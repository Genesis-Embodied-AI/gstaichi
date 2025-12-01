$ErrorActionPreference = "Stop"
Set-PSDebug -Trace 1
trap { Write-Error $_; exit 1 }

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$HOME\.local\bin;$env:Path"
uv venv

# This will install Visual Studio Build Tools, then exit, with an (intentional) exception:
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "build.py" -ErrorAction SilentlyContinue -Wait
