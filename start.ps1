# ============================================
# 🚀 EpigrafIA - Start Script
# ============================================
# Inicia ambos servidores: Backend (Python) y Frontend (Astro)

Write-Host "🚀 Iniciando EpigrafIA..." -ForegroundColor Cyan
Write-Host ""

# Matar procesos anteriores si existen
Write-Host "🧹 Limpiando procesos anteriores..." -ForegroundColor Yellow
taskkill /F /IM node.exe 2>$null
taskkill /F /IM python.exe 2>$null
Start-Sleep -Seconds 1

# Directorio del proyecto
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Iniciar Backend (Python FastAPI)
Write-Host "🐍 Iniciando Backend (Python FastAPI en puerto 8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir'; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 2

# Iniciar Frontend (Astro)
Write-Host "⚡ Iniciando Frontend (Astro en puerto 4321)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectDir\frontend'; npm run dev"

Start-Sleep -Seconds 3

# Abrir navegador
Write-Host ""
Write-Host "🌐 Abriendo navegador..." -ForegroundColor Cyan
Start-Process "http://localhost:4321"

Write-Host ""
Write-Host "✅ ¡EpigrafIA está corriendo!" -ForegroundColor Green
Write-Host "   Frontend: http://localhost:4321" -ForegroundColor White
Write-Host "   Backend:  http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "💡 Para detener, cierra las ventanas de PowerShell o ejecuta: taskkill /F /IM node.exe; taskkill /F /IM python.exe" -ForegroundColor DarkGray
