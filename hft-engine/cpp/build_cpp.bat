@echo off
REM Build the hft_cpp pybind11 module
REM Requires: cmake, Visual Studio 18 2026 (or newer), pybind11

setlocal enabledelayedexpansion

REM Find pybind11 cmake dir
for /f "tokens=*" %%i in ('python -c "import pybind11; print(pybind11.get_cmake_dir())"') do set PYBIND11_DIR=%%i

if exist build rmdir /s /q build

cmake -G "Visual Studio 18 2026" -A x64 -S . -B build ^
    -DCMAKE_BUILD_TYPE=Release ^
    -Dpybind11_DIR="%PYBIND11_DIR%"

if %errorlevel% neq 0 exit /b %errorlevel%

cmake --build build --config Release

if %errorlevel% equ 0 (
    echo.
    echo Success! Built: ..\src\hft_engine\hft_cpp.cp*-win_amd64.pyd
)
