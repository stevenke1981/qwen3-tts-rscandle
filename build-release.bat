@echo off
REM ===============================================================
REM build-release.bat — Build qwen3-tts GUI in release mode (CPU)
REM ===============================================================

set VS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools

REM Set up VS dev environment (needed for C/C++ compiler)
call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64

echo [build-release] Building GUI (release, CPU)...
echo.

cd /d "%~dp0"

cargo build --release --features gui --bin gui %*
