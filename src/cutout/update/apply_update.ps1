# Applies a downloaded Cutout update once the app has closed, then restarts it.
#
# Written by cutout.update.client: the plan JSON lists files to move in (already
# downloaded, unpacked and hash-checked) and old files to remove. Every file it
# replaces or removes is moved to a backup folder first, so a failure part-way
# puts the old version back exactly. Runs with Windows PowerShell 5.1, which every
# Windows 10/11 PC has. Only moves files inside the install folder.
param([Parameter(Mandatory = $true)][string]$PlanPath)

$ErrorActionPreference = 'Stop'
$plan = Get-Content -LiteralPath $PlanPath -Raw -Encoding UTF8 | ConvertFrom-Json

function Log([string]$msg) {
    try { Add-Content -LiteralPath $plan.log -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $msg) -Encoding UTF8 } catch {}
}

function Move-File([string]$from, [string]$to) {
    # Antivirus scanners briefly lock freshly written files, so retry (~15 s by default).
    for ($i = 0; ; $i++) {
        try {
            $dir = [System.IO.Path]::GetDirectoryName($to)
            [void][System.IO.Directory]::CreateDirectory($dir)
            [System.IO.File]::Move($from, $to)
            return
        } catch {
            if ($i -ge $retries) { throw }
            Start-Sleep -Milliseconds 500
        }
    }
}

function Write-Result([bool]$ok, [string]$message) {
    $data = @{ ok = $ok; version = $plan.version; message = $message; time = (Get-Date).ToString('o'); log = $plan.log }
    [System.IO.File]::WriteAllText($plan.result, ($data | ConvertTo-Json))
}

$app = $plan.app_dir
$retries = if ($plan.retries) { [int]$plan.retries } else { 30 }
$backup = $plan.backup_dir
$backedUp = New-Object System.Collections.Generic.List[string]
$added = New-Object System.Collections.Generic.List[object]
$ok = $false
$message = ''

Log "update to $($plan.version): $(@($plan.add).Count) files in, $(@($plan.remove).Count) out"
if ($plan.wait_pid -gt 0) {
    Log "waiting for process $($plan.wait_pid) to close"
    for ($i = 0; $i -lt 240; $i++) {
        if (-not (Get-Process -Id $plan.wait_pid -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }
}

if ($plan.wait_pid -gt 0 -and (Get-Process -Id $plan.wait_pid -ErrorAction SilentlyContinue)) {
    $message = 'Cutout did not close, so nothing was changed.'
    Log $message
} else {
    try {
        $targets = @(@($plan.remove) + @($plan.add | ForEach-Object { $_.dest })) | Sort-Object -Unique
        foreach ($rel in $targets) {
            $cur = Join-Path $app $rel
            if (Test-Path -LiteralPath $cur -PathType Leaf) {
                Move-File $cur (Join-Path $backup $rel)
                $backedUp.Add($rel)
            }
        }
        foreach ($a in @($plan.add)) {
            $dest = Join-Path $app $a.dest
            Move-File $a.src $dest
            $added.Add($a)
        }
        $ok = $true
        Log "applied: $($added.Count) files in, $($backedUp.Count) old files moved out"
    } catch {
        $message = "The update could not be installed ($($_.Exception.Message)). Your previous version is still installed."
        Log "failed, rolling back: $($_.Exception.Message)"
        foreach ($a in $added) {
            try { Move-File (Join-Path $app $a.dest) $a.src } catch { Log "rollback: cannot take back $($a.dest)" }
        }
        foreach ($rel in $backedUp) {
            try { Move-File (Join-Path $backup $rel) (Join-Path $app $rel) } catch { Log "rollback: cannot restore $rel" }
        }
        Log "rolled back"
    }
}

if ($ok) {
    try { Remove-Item -LiteralPath $backup -Recurse -Force -ErrorAction Stop } catch { Log "could not delete backup: $($_.Exception.Message)" }
}
Write-Result $ok $message
if ($plan.restart) {
    try { Start-Process -FilePath $plan.restart -WorkingDirectory $app } catch { Log "restart failed: $($_.Exception.Message)" }
}
Log 'done'
