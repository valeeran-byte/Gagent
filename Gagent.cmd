@echo off
rem Launch Gagent CLI. Keep this file pure ASCII: the Chinese user profile path
rem comes from %USERPROFILE% at runtime, and pushd/popd restores the caller's cwd.
setlocal
set "GAGENT_PYTHON=C:\Python\envs\agent\python.exe"
set "GAGENT_PROJECT=%USERPROFILE%\PycharmProjects\agent"

if not exist "%GAGENT_PYTHON%" (
    echo Gagent: Python not found: %GAGENT_PYTHON%
    exit /b 1
)
if not exist "%GAGENT_PROJECT%\cli.py" (
    echo Gagent: project not found: %GAGENT_PROJECT%\cli.py
    exit /b 1
)

chcp 65001 >nul
pushd "%GAGENT_PROJECT%"
"%GAGENT_PYTHON%" -X utf8 cli.py %*
set "GAGENT_RC=%ERRORLEVEL%"
popd
endlocal & exit /b %GAGENT_RC%
