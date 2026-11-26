@echo off
cd /d "%~dp0"
set "OutputDir=%~dp0Output"

echo Checking output dir...
if not exist "%OutputDir%" mkdir "%OutputDir%"

echo Set paths...
set "PATH=%~dp0Build-Tools\w64devkit-x64-250\bin;%PATH%"

echo Compiling...
gcc ../src/ChimbaBenchCPU.c -O3 -fopenmp -Wall -static -o "%OutputDir%\ChimbaBenchCPU-Win-x86_64.exe"

if %errorlevel% equ 0 (
	echo Complete!
) else (
	echo Error
)
