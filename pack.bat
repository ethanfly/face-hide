@echo off
cd /d "%~dp0"
python pack\build.py
if errorlevel 1 pause
