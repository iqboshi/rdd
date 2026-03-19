@echo off
setlocal
set SCRIPT_DIR=%~dp0

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%backup-code-to-github.ps1" %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Backup failed with exit code %EXIT_CODE%.
  exit /b %EXIT_CODE%
)

echo [DONE] Backup command finished.
endlocal
