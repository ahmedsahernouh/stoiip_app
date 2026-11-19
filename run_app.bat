@echo off
cd /d "%~dp0"
streamlit run app.py --server.baseUrlPath "" --browser.serverAddress localhost --browser.serverPort 8501
