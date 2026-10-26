#!/usr/bin/env bash
#
# build_cpp.sh - build the hft_cpp pybind11 extension.
# Requires: a C++17 compiler, cmake >= 3.15, and pybind11 in the chosen
# interpreter. The built module is written to hft-engine/src/hft_engine/ so that
# `import hft_engine.hft_cpp` picks it up without touching sys.path.
#
# WHY THE INTERPRETER IS RESOLVED BEFORE `cd`:
#   Callers commonly pass a project-relative path (PYTHON=.venv/bin/python).
#   This script changes into its own directory, so resolving that path
#   afterwards would look for hft-engine/cpp/.venv/bin/python and fail.

set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
if [ -x "$PWD/$PYTHON_BIN" ]; then
    PYTHON_BIN="$PWD/$PYTHON_BIN"
elif command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "$PYTHON_BIN")"
else
    echo "Interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

cd "$(dirname "$0")"

# cmake is often installed into the same virtual environment as pybind11, and
# that directory is not necessarily on PATH in non-interactive shells.
PY_BIN_DIR="$(dirname "$PYTHON_BIN")"
case ":$PATH:" in
    *":$PY_BIN_DIR:"*) ;;
    *) PATH="$PY_BIN_DIR:$PATH" ;;
esac

if ! "$PYTHON_BIN" -c "import pybind11" >/dev/null 2>&1; then
    echo "pybind11 is not installed for $PYTHON_BIN. Run: $PYTHON_BIN -m pip install pybind11" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake not found. Run: $PYTHON_BIN -m pip install cmake" >&2
    exit 1
fi

PYBIND11_DIR="$("$PYTHON_BIN" -c 'import pybind11; print(pybind11.get_cmake_dir())')"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$PYTHON_BIN" \
    -DPYBIND11_FINDPYTHON=ON \
    -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build --parallel

echo "==> built:"
ls -1 ../src/hft_engine/hft_cpp*.so 2>/dev/null || ls -1 ../src/hft_engine/hft_cpp*.pyd 2>/dev/null || true
