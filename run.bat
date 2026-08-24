@echo off
cd /d "%~dp0"
python -m facehide %*
if errorlevel 1 pause
