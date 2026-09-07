@echo off
REM One-command end-to-end verification for AgentForge on Windows.
REM
REM   scripts\verify_all.bat "C:\path\to\Vinoth_N_Package_v1"
REM
REM The workspace is COPIED to a temp folder first, so nothing the agent
REM writes can touch your real resumes. Requires Ollama running with at
REM least one model installed.

setlocal
cd /d "%~dp0.."

if "%~1"=="" (
  echo Usage: scripts\verify_all.bat "C:\path\to\resume\folder" [deny^|ask^|allow]
  exit /b 2
)

set "WS=%~1"
set "PERM=%~2"
if "%PERM%"=="" set "PERM=allow"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Using interpreter: %PY%
"%PY%" scripts\verify_all.py --dir "%WS%" --permission %PERM%
set "RC=%ERRORLEVEL%"

if "%RC%"=="0" (
  echo.
  echo VERIFICATION PASSED
) else (
  echo.
  echo VERIFICATION FAILED with code %RC%
)
exit /b %RC%
