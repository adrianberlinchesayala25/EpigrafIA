@echo off
REM ============================================
REM 🚀 EpigrafIA - Start Script (Batch)
REM ============================================
REM Doble clic en este archivo para iniciar EpigrafIA

echo.
echo  ███████╗██████╗ ██╗ ██████╗ ██████╗  █████╗ ███████╗██╗ █████╗ 
echo  ██╔════╝██╔══██╗██║██╔════╝ ██╔══██╗██╔══██╗██╔════╝██║██╔══██╗
echo  █████╗  ██████╔╝██║██║  ███╗██████╔╝███████║█████╗  ██║███████║
echo  ██╔══╝  ██╔═══╝ ██║██║   ██║██╔══██╗██╔══██║██╔══╝  ██║██╔══██║
echo  ███████╗██║     ██║╚██████╔╝██║  ██║██║  ██║██║     ██║██║  ██║
echo  ╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝
echo.
echo  Detectando idioma y acento con IA
echo.
echo ============================================

cd /d "%~dp0"

echo [1/3] Limpiando procesos anteriores...
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM python.exe >nul 2>&1
timeout /t 1 >nul

echo [2/3] Iniciando Backend (Python en puerto 8000)...
start "EpigrafIA Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
timeout /t 3 >nul

echo [3/3] Iniciando Frontend (Astro en puerto 4321)...
start "EpigrafIA Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"
timeout /t 4 >nul

echo.
echo Abriendo navegador...
start http://localhost:4321

echo.
echo ============================================
echo  EpigrafIA esta corriendo!
echo  Frontend: http://localhost:4321
echo  Backend:  http://localhost:8000
echo ============================================
echo.
echo Presiona cualquier tecla para cerrar esta ventana...
echo (Los servidores seguiran corriendo)
pause >nul
