# ============================================
# EpigrafIA - Start Script
# ============================================
# Solo ejecuta este archivo y se abre todo.

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host ""
Write-Host "  Iniciando EpigrafIA..." -ForegroundColor Cyan
Write-Host ""

# Matar procesos anteriores de EpigrafIA si existen (solo en los puertos usados)
Write-Host "  Limpiando puertos 8000 y 4321..." -ForegroundColor Yellow
$ports = @(8000, 4321)
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 1

# Iniciar Backend (Python FastAPI)
Write-Host "  Iniciando Backend (FastAPI en puerto 8000)..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$projectDir'; Write-Host 'Backend EpigrafIA - Puerto 8000' -ForegroundColor Green; Write-Host ''; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 2

# Iniciar Frontend (Astro)
Write-Host "  Iniciando Frontend (Astro en puerto 4321)..." -ForegroundColor Magenta
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "Set-Location '$projectDir\frontend'; Write-Host 'Frontend EpigrafIA - Puerto 4321' -ForegroundColor Magenta; Write-Host ''; npm run dev"

Start-Sleep -Seconds 3

# Abrir navegador
Write-Host ""
Write-Host "  Abriendo navegador..." -ForegroundColor Cyan
Start-Process "http://localhost:4321"

Write-Host ""
Write-Host "  EpigrafIA esta corriendo!" -ForegroundColor Green
Write-Host "    Frontend: http://localhost:4321" -ForegroundColor White
Write-Host "    Backend:  http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "  Para detener, cierra las ventanas de PowerShell abiertas." -ForegroundColor DarkGray
Write-Host ""
