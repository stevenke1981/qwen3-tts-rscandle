@echo off
REM ===============================================================
REM build-cuda.bat — Build qwen3-tts with CUDA support
REM 
REM Prerequisites:
REM   1. Visual Studio 2026 Build Tools (or VS 2026) installed
REM   2. CUDA 13.2 installed at default path
REM   3. Run this from any command prompt (it sets up VS + CUDA env)
REM ===============================================================

set VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2

REM Set up VS dev environment
call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64

REM Override CUDA_PATH to v13.2 (system default may point to v12.1)
set CUDA_PATH=%CUDA_ROOT%
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_ROOT%
set PATH=%CUDA_ROOT%\bin;%PATH%

echo [build-cuda.bat] CUDA_PATH=%CUDA_PATH%
echo [build-cuda.bat] Building qwen3-tts with CUDA support...
echo.

cd /d "%~dp0"

cargo build --features cli,cuda %*
