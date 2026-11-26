@echo off
rem Stops the llama-server that start.bat launched, by the port it listens on.
rem
rem Usage:  stop.bat             port 8080
rem         stop.bat 8081        another port
setlocal

set "PORT=%~1"
if "%PORT%"=="" set "PORT=%INK_PORT%"
if "%PORT%"=="" set "PORT=8080"

set "STOPPED="
rem netstat's fifth column is the owning PID. Only the server on this port is touched; a llama-server serving
rem something else, or on another port, is left alone.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
    if not errorlevel 1 (
        echo stopped llama-server, PID %%p, port %PORT%
        set "STOPPED=1"
    )
)

if not defined STOPPED echo nothing is listening on port %PORT%

ping -n 2 127.0.0.1 >nul
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 echo WARNING: port %PORT% still listens
endlocal
exit /b 0