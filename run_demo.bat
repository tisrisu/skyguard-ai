@echo off
rem Starts the dashboard and opens it in the browser.
rem From PowerShell run it as .\run_demo.bat
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

rem headless skips Streamlit's first-run email prompt, so open the browser ourselves
start "" /min cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
python -m streamlit run dashboard\app.py --server.headless true --server.port 8501
