# build_cpp.ps1 - build the hft_cpp pybind11 extension on Windows.
# Requires: Visual Studio Build Tools (C++ workload), cmake >= 3.15,
#           and `pip install pybind11` into the active interpreter.
#
# The generator is NOT hardcoded: cmake picks the newest installed Visual Studio
# or falls back to whatever default generator is configured, so this works on
# VS 2019 / 2022 / 2026 without edits.

param(
    [string]$Config = "Release",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $PythonExe) {
    $PythonExe = (Get-Command python).Source
}

& $PythonExe -c "import pybind11" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "pybind11 is not installed for $PythonExe. Run: $PythonExe -m pip install pybind11"
}

$pybind11Dir = (& $PythonExe -c "import pybind11; print(pybind11.get_cmake_dir())").Trim()

cmake -S . -B build -DPython_EXECUTABLE="$PythonExe" -DPYBIND11_FINDPYTHON=ON -Dpybind11_DIR="$pybind11Dir"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

cmake --build build --config $Config --parallel
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "==> built:"
Get-ChildItem ..\src\hft_engine\hft_cpp*.pyd | Select-Object -ExpandProperty Name
