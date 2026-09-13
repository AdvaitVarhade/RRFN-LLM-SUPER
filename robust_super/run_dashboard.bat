@echo off
echo =====================================================================
echo Launching RRFN-LLM-SUPER Interactive Simulation Dashboard...
echo =====================================================================
streamlit run robust_super/app.py --server.port 8501 --server.headless false
pause
