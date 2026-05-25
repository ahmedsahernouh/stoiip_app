0<0# : ^
"""
@echo off
REM ================================
REM   Batch section
REM ================================
echo Starting...

REM Check for Python in PATH
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not in the PATH. Please install Python or add it to PATH.
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo Error: Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing required Python packages...
REM NOTE: packages must be space-separated, NOT semicolon-separated
python -m pip install --upgrade numpy pandas plotly pyvista zmapio scipy altair streamlit
if %errorlevel% neq 0 (
    echo Error: Failed to install required Python packages.
    pause
    exit /b 1
)

echo.
echo Running Python script part...
python "%~f0" %*
set EXITCODE=%ERRORLEVEL%

echo.
echo Python script finished with exit code %EXITCODE%.
pause
exit /b %EXITCODE%
"""
# ================================
#   Python section
# ================================

