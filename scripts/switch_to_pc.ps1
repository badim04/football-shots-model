<#
Move the scheduled job from the Vultr server back to this PC.

    powershell -ExecutionPolicy Bypass -File scripts\switch_to_pc.ps1

If the server still answers: stops its timer, waits for a run in progress,
and copies its database and model back here (the PC's old copy is kept
as data\matches.db.bak-<date>). Then turns the PC's scheduled task back on.

If the server is already gone (subscription ended), the PC carries on from
its own copy. The next run fetches every result it missed, so predictions
are fine. What it doesn't know is what the server posted to Telegram since
the last switch, so the current block may be posted again, and results for
blocks the server posted won't be sent.
#>
param([string]$Server = "root@199.247.2.139")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$App = "/opt/football-shots"
$TaskName = "Football shots daily"
Set-Location $Repo

Write-Host "== Checking the server"
ssh -o BatchMode=yes -o ConnectTimeout=15 $Server "true" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "== Stopping the server's timer (waiting for a run in progress)"
    ssh -o BatchMode=yes $Server "systemctl disable -q --now football-shots.timer; while systemctl is-active -q football-shots.service; do sleep 5; done"
    $stamp = Get-Date -Format "yyyyMMdd-HHmm"
    if (Test-Path data\matches.db) { Copy-Item data\matches.db "data\matches.db.bak-$stamp" }
    Write-Host "== Copying the server's database and model here"
    scp -q "${Server}:$App/data/matches.db" data\matches.db
    if ($LASTEXITCODE -ne 0) { throw "Couldn't copy the database - the PC task was NOT turned on." }
    scp -q "${Server}:$App/models/model_bundle.joblib" models\model_bundle.joblib
} else {
    Write-Warning "Server not reachable - carrying on with the PC's own copy of the data."
}

Write-Host "== Turning the PC's scheduled task back on"
Enable-ScheduledTask -TaskName $TaskName | Out-Null
Write-Host ""
Write-Host "Done - the PC runs the job every 2 hours again (log: logs\daily.log)."
