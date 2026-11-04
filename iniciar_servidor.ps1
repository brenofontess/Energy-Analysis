# Iniciar Servidor do Workflow Canvas
Set-Location -Path $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "INICIANDO SERVIDOR DO WORKFLOW CANVAS (127.0.0.1:8765)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$pythonCandidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "python.exe"
)

$pythonExe = $null
foreach ($cand in $pythonCandidates) {
    if (Test-Path $cand) {
        $pythonExe = $cand
        break
    }
}

if (-not $pythonExe) {
    Write-Host "[ERRO] Executavel Python nao encontrado." -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[INFO] A utilizar Python: $pythonExe" -ForegroundColor Green
Write-Host "[INFO] A abrir http://127.0.0.1:8765/ no navegador..." -ForegroundColor Green
Write-Host "[INFO] Pressione Ctrl+C para encerrar o servidor." -ForegroundColor Yellow
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray

Start-Process "http://127.0.0.1:8765/"
& $pythonExe scripts/canvas_server.py
