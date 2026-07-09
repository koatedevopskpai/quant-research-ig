@echo off
REM Build the C++ pybind11 accelerator module.
REM Requires: Visual Studio Build Tools, CMake, pybind11 (pip install pybind11)
REM
REM Usage: from the hft-engine root: cpp\build_cpp.bat
REM   or:  cd cpp && build_cpp.bat

REM Find VS - try common locations
if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul
) else (
    echo Error: Visual Studio Build Tools not found
    exit /b 1
)

REM Find pybind11 cmake dir
for /f %%i in ('python -c "import pybind11; print(pybind11.get_cmake_dir())" 2^>nul') do set PYBIND11_DIR=%%i
if "%PYBIND11_DIR%"=="" (
    echo Error: pybind11 not found
    exit /b 1
)

cmake -S . -B build -A x64 -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="%PYBIND11_DIR%"
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

cmake --build build --config Release
