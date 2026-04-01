# run_all.ps1
# 루트 디렉터리에서 실행:  .\run_all.ps1
#
# 전제 조건
#   HandEye_Calibration\.venv37\  — Python 3.7 가상환경
#   VoiceTeaching\.venv311\       — Python 3.11 가상환경
#
# VoiceTeaching 이 action.json 을 아래 경로에 쓴다고 가정:
#   VoiceTeaching\assets\action.json

$ROOT = $PSScriptRoot

# ── VoiceTeaching\assets 디렉터리 보장 ─────────────────────────────
$assetsDir = Join-Path $ROOT "VoiceTeaching\assets"
if (-not (Test-Path $assetsDir)) {
    New-Item -ItemType Directory -Path $assetsDir | Out-Null
    Write-Host "[run_all] Created: $assetsDir"
}

# ── 각 프로젝트의 Python 인터프리터 경로 ───────────────────────────
$py37  = Join-Path $ROOT "HandEye_Calibration\.venv37\Scripts\python.exe"
$py311 = Join-Path $ROOT "VoiceTeaching\.venv311\Scripts\python.exe"

# 인터프리터 존재 확인
foreach ($py in @($py37, $py311)) {
    if (-not (Test-Path $py)) {
        Write-Error "[run_all] Python interpreter not found: $py"
        exit 1
    }
}

# ── HandEye_Calibration 실행 (Python 3.7) ──────────────────────────
Write-Host "[run_all] Starting HandEye_Calibration (Python 3.7)..."
$navJob = Start-Process -FilePath $py37 `
    -ArgumentList "-m", "src.main" `
    -WorkingDirectory (Join-Path $ROOT "HandEye_Calibration") `
    -PassThru `
    -NoNewWindow

# ── VoiceTeaching 실행 (Python 3.11) ───────────────────────────────
Write-Host "[run_all] Starting VoiceTeaching (Python 3.11)..."
$voiceJob = Start-Process -FilePath $py311 `
    -ArgumentList "-m", "src.main" `
    -WorkingDirectory (Join-Path $ROOT "VoiceTeaching") `
    -PassThru `
    -NoNewWindow

Write-Host ""
Write-Host "[run_all] Both processes started."
Write-Host "  HandEye_Calibration PID : $($navJob.Id)"
Write-Host "  VoiceTeaching       PID : $($voiceJob.Id)"
Write-Host ""
Write-Host "Ctrl+C 또는 이 창을 닫으면 두 프로세스가 종료됩니다."

# ── 두 프로세스 중 하나라도 종료되면 나머지도 정리 ────────────────
try {
    while (-not $navJob.HasExited -and -not $voiceJob.HasExited) {
        Start-Sleep -Milliseconds 500
    }
} finally {
    foreach ($proc in @($navJob, $voiceJob)) {
        if (-not $proc.HasExited) {
            Write-Host "[run_all] Stopping PID $($proc.Id)..."
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "[run_all] All processes stopped."
}