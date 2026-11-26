@echo off
rem One-line launcher for llama-server: Inkling 2 Mini with the DSpark drafter, in the best configuration measured
rem on this machine on 2026-11-26 -- 131.27 tok/s writing at context 65536 (56,058-token prompt, 92 % of the drafts
rem accepted). Same build and the same command line as the README section; only paths, port and context are variables.
rem
rem Usage:  start.bat                  context 65536, port 8080
rem         start.bat 131072           another context
rem         start.bat 131072 8081      another context and port
rem         set INK_MODEL=... & set INK_DRAFT=...   before start.bat to use other files
setlocal

set "CTX=%~1"
set "PORT=%~2"
if "%CTX%"==""  set "CTX=%INK_CTX%"
if "%CTX%"==""  set "CTX=65536"
if "%PORT%"=="" set "PORT=%INK_PORT%"
if "%PORT%"=="" set "PORT=8080"

set "HERE=%~dp0"
for %%d in ("%HERE%..\..") do set "ROOT=%%~fd"
set "EXE=%ROOT%\build-hip\bin\llama-server.exe"
set "RUN=%ROOT%\run"
set "LOG=%RUN%\llama-server.log"
set "ERRLOG=%RUN%\llama-server.err.log"

if defined INK_MODEL (set "MODEL=%INK_MODEL%") else (set "MODEL=D:\software\llama_cpp\model\inkling2\Inkling-2-Mini-UD-Q4_K_XL-old.gguf")
if defined INK_DRAFT (set "DRAFT=%INK_DRAFT%") else (set "DRAFT=D:\software\llama_cpp\model\inkling2\dspark-Inkling-2-Mini-Q8_0_Old.gguf")

if not exist "%EXE%" (
    echo ERROR: no llama-server at "%EXE%"
    echo        build it first: scripts\inkling2-amd\build-hip.cmd
    exit /b 1
)
if not exist "%MODEL%" (
    echo ERROR: model not found: "%MODEL%"
    echo        set INK_MODEL to the Inkling 2 GGUF with the old 'inkling2' architecture
    echo        a file that says 'inkling2_loop' will not load on this branch
    exit /b 1
)
if not exist "%DRAFT%" (
    echo ERROR: drafter not found: "%DRAFT%"
    echo        set INK_DRAFT to dspark-Inkling-2-Mini-Q8_0_Old.gguf
    exit /b 1
)
if not exist "%RUN%" mkdir "%RUN%"

netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo ERROR: port %PORT% is already in use; run stop.bat first
    exit /b 1
)

rem -dev ROCm0 has to be the RX 7900 XTX: this machine also enumerates the Ryzen 780M next to it, and the model
rem does not fit on the iGPU. Ask the build first rather than trusting the index.
"%EXE%" --list-devices > "%RUN%\devices.txt" 2>&1
findstr /c:"ROCm0:" "%RUN%\devices.txt" >nul
if errorlevel 1 (
    echo ERROR: the build reports no ROCm0 device. Is the eGPU connected?
    type "%RUN%\devices.txt"
    exit /b 1
)
findstr /c:"ROCm0:" "%RUN%\devices.txt" | findstr /i "7900" >nul
if errorlevel 1 (
    echo WARNING: ROCm0 is not the RX 7900 XTX:
    findstr /c:"ROCm0:" "%RUN%\devices.txt"
    echo          the weights do not fit on the iGPU; reconnect the eGPU and run start.bat again
)

rem 1> holds the tokens and the sampling lines, 2> the log of llama.cpp itself (that is where the device and the
rem draft statistics go). The window is minimised; stop.bat closes it.
echo starting llama-server -- context %CTX%, port %PORT%
start "Inkling2 llama-server :%PORT%" /min cmd /c ""%EXE%" -m "%MODEL%" -dev ROCm0 -ngl 99 -fa 1 -c %CTX% -np 1 -b 2048 -ub 1024 --jinja -md "%DRAFT%" -ngld 99 --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-n-ubatch 512 --backend-sampling --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 --host 127.0.0.1 --port %PORT% 1>"%LOG%" 2>"%ERRLOG%""

rem The weights are 16.4 GiB and the drafter 384 MiB: a cold start needs about 30-60 s, so wait up to 240 s.
set /a TRIES=0
:wait
curl -sf -o nul --max-time 2 "http://127.0.0.1:%PORT%/health" >nul 2>&1
if not errorlevel 1 goto ready
set /a TRIES+=1
if %TRIES% GEQ 240 goto timeout
ping -n 2 127.0.0.1 >nul
goto wait

:ready
echo.
echo ready: http://127.0.0.1:%PORT%
echo stop with: stop.bat %PORT%
echo log:       %ERRLOG%
endlocal
exit /b 0

:timeout
echo ERROR: not healthy after 240 s. The log is %ERRLOG%
exit /b 1
