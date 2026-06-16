@echo off
REM ===============================================================
REM build-release-cuda.bat — Build qwen3-tts GUI release + CUDA
REM ===============================================================

set VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2

call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64

set CUDA_PATH=%CUDA_ROOT%
set CUDA_TOOLKIT_ROOT_DIR=%CUDA_ROOT%
set PATH=%CUDA_ROOT%\bin;%PATH%

echo [build-release-cuda] CUDA_PATH=%CUDA_PATH%
echo [build-release-cuda] Building GUI (release, CUDA)...
echo.

cd /d "%~dp0"

cargo build --release --features cli,cuda,gui %*
