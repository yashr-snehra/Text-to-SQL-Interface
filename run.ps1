# run.ps1 -- set up and launch the Text-to-SQL Interface.
#
#   .\run.ps1              full setup (venv + deps + model + sample db + tests) then run
#   .\run.ps1 -SkipSetup   just launch backend + frontend (everything already in place)
#
# Override the model with:  $env:OLLAMA_MODEL = "qwen2.5-coder:7b"   (before running)
param([switch]$SkipSetup)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$model = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen2.5-coder:14b" }

if (-not $SkipSetup) {
    # 1. Bootstrap interpreter: prefer the py launcher (real CPython), else python.
    $bootstrap = "python"; $bootArgs = @()
    if (Get-Command py -ErrorAction SilentlyContinue) { $bootstrap = "py"; $bootArgs = @("-3") }

    # 2. Virtual environment + dependencies
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating virtual environment (.venv)..."
        & $bootstrap @bootArgs -m venv .venv
    }
    Write-Host "Installing Python dependencies..."
    & $venvPython -m pip install --quiet --upgrade pip
    & $venvPython -m pip install --quiet -r requirements.txt

    # 3. Ollama model. If a server is already serving on :11434 (e.g. the Intel
    #    IPEX-LLM build, which isn't on PATH), use it as-is; otherwise pull via CLI.
    $serverUp = $false
    try { Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 -UseBasicParsing | Out-Null; $serverUp = $true } catch {}
    if ($serverUp) {
        Write-Host "Ollama server already running on :11434 -- make sure model '$model' is pulled there (ollama pull $model)."
    } elseif (Get-Command ollama -ErrorAction SilentlyContinue) {
        $have = (& ollama list | Out-String) -match [regex]::Escape($model)
        if ($have) {
            Write-Host "Model '$model' already present."
        } else {
            Write-Host "Pulling model '$model' (several GB, one-time)..."
            & ollama pull $model
        }
    } else {
        Write-Warning "No Ollama server on :11434 and 'ollama' not on PATH. Start your server (e.g. C:\ipex-ollama\start-ollama.bat) or install from https://ollama.com, then: ollama pull $model"
    }

    # 4. Demo database + self-check
    & $venvPython make_sample_db.py
    & $venvPython test_text2sql.py
}

if (-not (Test-Path $venvPython)) { throw "No .venv found. Run once without -SkipSetup first." }

# 5. Backend (background) + frontend (foreground). Ctrl+C stops both.
Write-Host "Starting FastAPI on http://localhost:8000 ..."
$api = Start-Process -PassThru -FilePath $venvPython -ArgumentList @("-m", "uvicorn", "api:app", "--port", "8000")
try {
    Start-Sleep -Seconds 2
    Write-Host "Starting Streamlit (Ctrl+C to stop) ..."
    & $venvPython -m streamlit run app.py
} finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}
