@echo off
REM Build the hft_cpp pybind11 module (Windows wrapper).
REM The real logic lives in build_cpp.ps1, which does not hardcode a Visual
REM Studio version. Requires: VS Build Tools (C++ workload), cmake, pybind11.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_cpp.ps1" %*
