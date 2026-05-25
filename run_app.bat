@echo off
REM Simple launcher for the STOIIP Streamlit app
setlocal
cd /d "%~dp0"
echo Starting Streamlit STOIIP app...
streamlit run app.py
endlocal
