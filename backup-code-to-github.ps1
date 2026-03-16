param(
    [string]$RepoUrl = "https://github.com/iqboshi/rdd.git",
    [string]$Branch = "master",
    [string]$CommitMessage = ""
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message"
}

$sourceRoot = Split-Path -Parent $PSCommandPath
$tempRepo = Join-Path $sourceRoot ".repo_publish_tmp"

if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
    $CommitMessage = "backup: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

Write-Step "Source root: $sourceRoot"
Write-Step "Commit message: $CommitMessage"

if (Test-Path $tempRepo) {
    Write-Step "Removing stale temp repo: $tempRepo"
    Remove-Item -Recurse -Force $tempRepo
}

try {
    Write-Step "Cloning remote repo"
    git clone $RepoUrl $tempRepo | Out-Host

    Push-Location $tempRepo
    try {
        Write-Step "Clearing tracked workspace (keep .git)"
        Get-ChildItem -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force

        $excludeDirs = @(
            (Join-Path $sourceRoot ".git"),
            (Join-Path $sourceRoot ".repo_publish_tmp"),
            (Join-Path $sourceRoot ".tmp"),
            (Join-Path $sourceRoot "__pycache__"),
            (Join-Path $sourceRoot "data"),
            (Join-Path $sourceRoot "outputs"),
            (Join-Path $sourceRoot "visual"),
            (Join-Path $sourceRoot "rdd-dev"),
            (Join-Path $sourceRoot "pretrain_riceseg\outputs")
        )
        $excludeFiles = @("*.pth", "*.pt", "*.ckpt", "*.onnx", "*.bin", "*.safetensors", "*.npy", "*.npz", "*.pyc")

        $robocopyArgs = @(
            "`"$sourceRoot`"",
            "`"$tempRepo`"",
            "/E",
            "/R:1",
            "/W:1"
        )
        foreach ($d in $excludeDirs) {
            $robocopyArgs += "/XD"
            $robocopyArgs += "`"$d`""
        }
        foreach ($f in $excludeFiles) {
            $robocopyArgs += "/XF"
            $robocopyArgs += $f
        }

        Write-Step "Copying code with robocopy (excluding data/weights/artifacts)"
        $robocopyCmd = "robocopy " + ($robocopyArgs -join " ")
        Invoke-Expression $robocopyCmd | Out-Host
        if ($LASTEXITCODE -gt 7) {
            throw "robocopy failed with exit code $LASTEXITCODE"
        }

        Write-Step "Cleaning bytecode caches"
        Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -File -Include *.pyc -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue

        @'
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual envs
.venv/
venv/
rdd-dev/

# Local temp/cache
.tmp/
*.log

# Data and experiment artifacts
data/
outputs/
visual/
pretrain_riceseg/outputs/

# Model weights / checkpoints
*.pth
*.pt
*.ckpt
*.onnx
*.bin
*.safetensors

# Numpy data artifacts
*.npy
*.npz

# OS/editor
.DS_Store
Thumbs.db
.vscode/
.idea/
'@ | Set-Content -Path ".gitignore" -Encoding UTF8

        Write-Step "Checking changes"
        git add -A
        $status = git status --porcelain
        if ([string]::IsNullOrWhiteSpace(($status | Out-String))) {
            Write-Host "[INFO] No code changes detected. Skip commit/push."
            exit 0
        }

        Write-Step "Committing changes"
        git commit -m $CommitMessage | Out-Host

        Write-Step "Pushing to remote"
        git push origin $Branch | Out-Host

        Write-Host "[DONE] Backup pushed successfully to $RepoUrl ($Branch)"
    }
    finally {
        Pop-Location
    }
}
finally {
    if (Test-Path $tempRepo) {
        Write-Step "Removing temp repo"
        Remove-Item -Recurse -Force $tempRepo
    }
}
