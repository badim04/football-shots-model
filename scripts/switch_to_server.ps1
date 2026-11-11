<#
Move the scheduled job from this PC to the Vultr server.

    powershell -ExecutionPolicy Bypass -File scripts\switch_to_server.ps1

Stops the PC's scheduled task, copies the committed code plus .env, the
database and the model to the server, and starts the server's
every-2-hours timer. Only one place runs at a time, so nothing is posted
twice.

If the server is already the active one, this only updates its code (your
latest commit) and leaves its data alone - the server's copy is newer.
#>
param([string]$Server = "root@199.247.2.139")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$App = "/opt/football-shots"
$TaskName = "Football shots daily"
Set-Location $Repo

function Remote($cmd) {
    ssh -o BatchMode=yes -o ConnectTimeout=15 $Server $cmd
    if ($LASTEXITCODE -ne 0) { throw "Server command failed: $cmd" }
}

Write-Host "== Checking the server"
$active = ssh -o BatchMode=yes -o ConnectTimeout=15 $Server "systemctl is-active football-shots.timer 2>/dev/null || true"
if ($LASTEXITCODE -ne 0) { throw "Can't reach $Server - is the Vultr subscription active?" }
$updateOnly = ($active -eq "active")

if (git status --porcelain -- src scripts deploy config.py) {
    Write-Warning "You have uncommitted code changes - the server gets your last commit, not them."
}

Write-Host "== Stopping the PC's scheduled task"
Disable-ScheduledTask -TaskName $TaskName | Out-Null
while ((Get-ScheduledTask -TaskName $TaskName).State -eq "Running") { Start-Sleep 5 }

Write-Host "== Copying code (last commit)"
$tar = Join-Path $env:TEMP "football-shots-code.tar"
git archive --format=tar -o $tar HEAD
Remote "mkdir -p $App/data $App/models"
scp -q $tar "${Server}:/tmp/football-shots-code.tar"
Remote "tar -xf /tmp/football-shots-code.tar -C $App && rm /tmp/football-shots-code.tar"
Remove-Item $tar

if ($updateOnly) {
    Write-Host "== Server is already active: code updated, its data left as it is"
    Remote "bash $App/deploy/setup.sh"
} else {
    Write-Host "== Copying .env, database and model"
    scp -q .env "${Server}:$App/.env"
    scp -q data\matches.db "${Server}:$App/data/matches.db"
    scp -q models\model_bundle.joblib "${Server}:$App/models/model_bundle.joblib"
    Write-Host "== Installing and starting the timer"
    Remote "bash $App/deploy/setup.sh start"
}

Write-Host ""
Write-Host "Done - the server runs the job now; the PC task is off."
Write-Host "Server log:  ssh $Server journalctl -u football-shots -n 50"
Write-Host "Back to PC:  powershell -ExecutionPolicy Bypass -File scripts\switch_to_pc.ps1"
