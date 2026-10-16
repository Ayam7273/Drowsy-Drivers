@echo off
REM One-click launcher: sets up the environment on first run, then starts the demo.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment...
    python -m venv .venv || (echo Python 3.12 not found. Install it from python.org & pause & exit /b 1)
)

if not exist ".venv\.deps_ok" (
    echo Installing dependencies - this takes a few minutes the first time...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Install failed & pause & exit /b 1)
    echo ok> ".venv\.deps_ok"
)

if not exist "models\yawn_rf.joblib" (
    echo NOTE: trained models not found in models\ - running with rules only.
    echo       To train them: see README, step "Train the models".
)

".venv\Scripts\python.exe" demo.py %*
if errorlevel 1 pause
