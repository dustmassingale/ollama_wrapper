@echo off
title Ollama Wrapper Proxy - Start

:: Ensure we're running from the script directory (so proxy.py is found when double-clicked)
pushd "%~dp0" >nul 2>&1

echo =========================================
echo Starting Ollama Wrapper Proxy
echo =========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not found in your PATH.
    echo.
    echo Please install Python from:
    echo   https://www.python.org/downloads/
    echo and ensure the 'python' command is available in your PATH.
    echo.
    echo Press any key to exit...
    pause >nul
    popd >nul 2>&1
    exit /b 1
)

:: Verify that proxy.py exists in the script directory
if not exist "proxy.py" (
    echo ERROR: proxy.py file not found in the current directory: %CD%
    echo Make sure start.bat is placed next to proxy.py, or update the script to point to the correct location.
    echo.
    echo Press any key to exit...
    pause >nul
    popd >nul 2>&1
    exit /b 1
)

echo Python detected! Launching proxy.py...
echo (Use Ctrl+C in this window to stop the proxy)
echo.
:: Install dependencies if a requirements.txt file exists
if exist "requirements.txt" (
    echo Found requirements.txt — installing dependencies...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo WARNING: Installing dependencies failed. You may need to run:
        echo   python -m pip install -r requirements.txt
        echo Press any key to continue and attempt to run proxy anyway, or Ctrl+C to abort.
        pause >nul
    ) else (
        echo Dependencies installed.
    )
) else (
    echo requirements.txt not found; skipping dependency installation.
)

:: Run the proxy
python proxy.py
set "RC=%ERRORLEVEL%"

echo.
echo proxy.py exited with code %RC%.
echo.

echo Press any key to exit...
pause >nul

popd >nul 2>&1
exit /b %RC%
