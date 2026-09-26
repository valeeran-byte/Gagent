@echo off
setlocal
chcp 65001 >nul

py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 goto python311

py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 goto pylauncher

python -c "import sys" >nul 2>&1
if not errorlevel 1 goto pythonpath

echo Gagent: Python 3.11 or newer is required.
exit /b 1

:python311
py -3.11 -X utf8 "%~dp0useorcreate.py" %*
exit /b %errorlevel%

:pylauncher
py -3 -X utf8 "%~dp0useorcreate.py" %*
exit /b %errorlevel%

:pythonpath
python -X utf8 "%~dp0useorcreate.py" %*
exit /b %errorlevel%
