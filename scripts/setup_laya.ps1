$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$environmentDirectory = Join-Path $repository '.local\laya-env'
$pythonExecutable = Join-Path $environmentDirectory 'Scripts\python.exe'
Push-Location -LiteralPath $repository
try {
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        uv venv --python 3.13 $environmentDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Laya environment' }
    }
    uv pip install --python $pythonExecutable -r requirements-laya.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
    if ($LASTEXITCODE -ne 0) { throw 'Could not install pinned Laya dependencies' }
    uv pip install --python $pythonExecutable --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Could not install the Zelda pilot commands' }
    Write-Output "Laya pilot interpreter: $pythonExecutable"
} finally {
    Pop-Location
}
