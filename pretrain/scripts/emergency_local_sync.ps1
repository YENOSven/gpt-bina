param(
    [ValidateSet("Pull", "Push")]
    [string]$Direction = "Pull",
    [string]$RemoteName = "gdrive",
    [string]$DrivePath = "Bina_Pretrain",
    [string]$LocalRoot = "C:\Bina_Emergency_Local",
    [string]$Stage = "pretrain",
    [switch]$IncludeCorpus
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

New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null

# $LocalRoot mirrors Drive's own layout exactly (LOCAL_FALLBACK_ROOT in
# phase3_pretrain_training.ipynb becomes DRIVE_ROOT directly, so the paths it expects --
# "$DRIVE_ROOT/$Stage/latest.pt", "$DRIVE_ROOT/corpus_train.bin" -- have to already look like
# real Drive paths locally, not some other structure).
if ($Direction -eq "Pull") {
    $source = "$($RemoteName):$DrivePath/$Stage"
    rclone lsf "$source" *>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "nothing at $source yet -- no checkpoint exists for stage '$Stage' on Drive. Nothing to pull."
        exit 1
    }

    Write-Host "pulling checkpoint: $source -> $LocalRoot\$Stage"
    rclone copy "$source" "$LocalRoot\$Stage" --progress

    if ($IncludeCorpus) {
        Write-Host "`npulling corpus (large, tens of GB -- this may take a while depending on your connection)"
        rclone copy "$($RemoteName):$DrivePath" "$LocalRoot" --include "corpus_*.bin" --progress
    }
    else {
        Write-Host "`nskipping corpus pull (-IncludeCorpus not set). Training needs corpus_train.bin present at "
        Write-Host "$LocalRoot\corpus_train.bin to actually run -- already there via a previous pull or Drive "
        Write-Host "Desktop sync? if not, re-run with -IncludeCorpus."
    }

    Write-Host "`nDone. In phase3_pretrain_training.ipynb's Setup cell, set:"
    Write-Host "  LOCAL_FALLBACK_ROOT = `"$LocalRoot`""
    Write-Host "and run the notebook locally (not in Colab)."
}
else {
    $dest = "$($RemoteName):$DrivePath/$Stage"
    $localStageDir = "$LocalRoot\$Stage"
    if (-not (Test-Path "$localStageDir\latest.pt")) {
        Write-Error "no $localStageDir\latest.pt found -- nothing to push. Did a local fallback session actually run and checkpoint?"
        exit 1
    }

    Write-Host "pushing local checkpoint: $localStageDir -> $dest"
    rclone copy "$localStageDir" "$dest" --progress

    Write-Host "`nDone. Drive now has this session's real progress -- safe to run a Colab session again; "
    Write-Host "it will resume from here, not from an older checkpoint."
}
