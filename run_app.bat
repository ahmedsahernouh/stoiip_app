@echo off
REM temporary place holder
setlocal
cd /d "%~dp0"
echo Starting Streamlit STOIIP app...
streamlit run app.py
endlocal
