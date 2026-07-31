param(
    [string]$RemoteName = "gdrive",
    [string]$DrivePath = "Bina_Pretrain",
    [string]$LocalRoot = "C:\Models",
    [string[]]$Stages = @("pretrain", "sft", "dpo")
)

$ErrorActionPreference = "Stop"

$rcloneCmd = Get-Command rclone -ErrorAction SilentlyContinue
if (-not $rcloneCmd) {
    Write-Error "rclone not found on PATH. It's installed (winget install Rclone.Rclone) but needs a fresh terminal to pick up PATH. Then run 'rclone config' once -- interactive, opens a browser for Google Drive auth -- and name the remote '$RemoteName' (or pass -RemoteName to match whatever you named it)."
    exit 1
}

$remotes = rclone listremotes
if ($remotes -notcontains "$($RemoteName):") {
    Write-Error "rclone remote '$RemoteName' isn't configured yet. Run 'rclone config' once (interactive, opens a browser for Google Drive auth) before using this script. Configured remotes found: $($remotes -join ', ')"
    exit 1
}

# Only milestone checkpoints get mirrored -- not the constantly-overwritten latest.pt, which
# changes every 15-30 minutes during a training session and would just burn bandwidth for no
# benefit copying locally that often. Run this manually after a session, or on whatever
# schedule you like via Task Scheduler.
foreach ($stage in $Stages) {
    $source = "$($RemoteName):$DrivePath/$stage/milestones"
    $dest = Join-Path $LocalRoot "bina-pretrain-$stage\milestones"

    rclone lsf "$source" *>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "no milestones yet for stage '$stage' (nothing at $source) -- skipping"
        continue
    }

    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Write-Host "mirroring $source -> $dest"
    rclone copy "$source" "$dest" --update --progress
}

Write-Host "`nDone. Milestone checkpoints are mirrored under $LocalRoot -- latest.pt stays Drive-only by design."
