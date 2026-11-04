@echo off
title Servidor Canvas - Energy Price Analysis
cd /d "%~dp0"

echo ======================================================================
echo INICIANDO SERVIDOR DO WORKFLOW CANVAS (127.0.0.1:8765)
echo ======================================================================

set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if exist "%PY_EXE%" goto RUN

set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if exist "%PY_EXE%" goto RUN

set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%PY_EXE%" goto RUN

set "PY_EXE=python"

:RUN
echo Utilizando Python: %PY_EXE%
echo Aceda no navegador a: http://127.0.0.1:8765/
echo Pressione Ctrl+C nesta janela para parar o servidor.
echo ----------------------------------------------------------------------
timeout /t 2 >nul
start http://127.0.0.1:8765/
"%PY_EXE%" scripts\canvas_server.py
pause
