$ErrorActionPreference = 'Stop'

$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Snmf = (Resolve-Path (Join-Path $Project 'external\snmf')).Path
$PathParts = @($Project, $Snmf)
if ($env:PYTHONPATH) { $PathParts += $env:PYTHONPATH }
$env:PYTHONPATH = $PathParts -join [System.IO.Path]::PathSeparator
$Python = (Get-Command python -ErrorAction Stop).Source
$Concept = 'Pornography'

function Invoke-RealFlow {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Config
    )
    $ConfigPath = (Resolve-Path (Join-Path $Project "configs\$Config")).Path
    Write-Host "[REAL FLOW] $Name"
    Write-Host "config=$ConfigPath"
    & $Python -m ember.real_flow_test `
        --config $ConfigPath `
        --concept $Concept `
        --execution windows
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Set-Location $Project
& $Python -c "import torch, ember, factorization; print('python=',r'$Python'); print('cuda=',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
if ($LASTEXITCODE -ne 0) { throw 'Python environment preflight failed' }

Invoke-RealFlow `
    -Name 'Windows real CPU model + CPU feature fitting' `
    -Config 'ember_lment_real_windows_cpu.yaml'

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA is required for the Windows GPU real-flow test'"
if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }

Invoke-RealFlow `
    -Name 'Windows real CUDA model + CUDA feature fitting' `
    -Config 'ember_lment_real_windows_gpu.yaml'

Write-Host '[PASS] Windows real CPU and GPU flows completed.'
Write-Host "Retained outputs: $Project\runs"
