@echo off
setlocal
echo === SyncGuard AI: one-time setup ===
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://www.python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
if not exist venv (
  python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete. Double-click run_app.bat to start the dashboard.
pause
