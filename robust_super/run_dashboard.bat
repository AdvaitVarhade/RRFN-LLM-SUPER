@echo off
cd /d "%~dp0\.."
echo =====================================================================
echo Launching RRFN-LLM-SUPER Interactive Simulation Dashboard...
echo =====================================================================
python -m streamlit run robust_super/app.py --server.port 8501
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit exited with error. If streamlit is not installed, run:
    echo pip install -r requirements.txt
    echo.
)
pause
