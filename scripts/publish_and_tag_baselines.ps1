$ErrorActionPreference = 'Stop'
$sourceRoot = 'D:\py_project\rdd-3-6'
$tempRepo = Join-Path $sourceRoot ('.repo_tag_tmp_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$repoUrl = 'https://github.com/iqboshi/rdd.git'
$branch = 'master'
$tagA4 = 'baseline-a4-r3-20260318'
$tagA7 = 'baseline-a7-s2-20260318'
$commitMessage = 'baseline: switch fixed baseline to A7-s2 + blind decode (2026-03-18)'

function Invoke-GitNoProxy {
  param([Parameter(Mandatory=$true)][string]$GitArgs)
  $cmd = "git -c http.proxy= -c https.proxy= $GitArgs"
  Write-Host "[GIT] $cmd"
  Invoke-Expression $cmd
}

Invoke-GitNoProxy "clone $repoUrl $tempRepo"
Push-Location $tempRepo
try {
  Invoke-GitNoProxy "checkout $branch"
  Invoke-GitNoProxy "pull origin $branch"

  $oldHead = (git rev-parse HEAD).Trim()
  Write-Host "[INFO] Old HEAD (A4-r3 tag target): $oldHead"

  $a4RemoteExists = -not [string]::IsNullOrWhiteSpace((Invoke-GitNoProxy "ls-remote --tags origin refs/tags/$tagA4" | Out-String))
  $a4TaggedNow = $false
  if (-not $a4RemoteExists) {
    git tag -a $tagA4 $oldHead -m "A4-r3 fixed-baseline snapshot before switching to A7-s2 (2026-03-18)"
    $a4TaggedNow = $true
    Write-Host "[INFO] Created tag: $tagA4 -> $oldHead"
  } else {
    Write-Host "[INFO] Remote tag already exists, skip create: $tagA4"
  }

  $toRemove = Get-ChildItem -Force | Where-Object { $_.Name -ne '.git' }
  foreach ($item in $toRemove) {
    $removed = $false
    for ($i = 0; $i -lt 8 -and -not $removed; $i++) {
      try {
        Remove-Item -Recurse -Force $item.FullName -ErrorAction Stop
        $removed = $true
      } catch {
        Start-Sleep -Milliseconds 300
      }
    }
    if (-not $removed) {
      Write-Host "[WARN] Skip locked path: $($item.FullName)"
    }
  }

  $excludeDirs = @(
    (Join-Path $sourceRoot '.git'),
    (Join-Path $sourceRoot '.repo_publish_tmp'),
    (Join-Path $sourceRoot '.repo_tag_tmp'),
    (Join-Path $sourceRoot '.tmp'),
    (Join-Path $sourceRoot '__pycache__'),
    (Join-Path $sourceRoot 'data'),
    (Join-Path $sourceRoot 'outputs'),
    (Join-Path $sourceRoot 'visual'),
    (Join-Path $sourceRoot 'rdd-dev'),
    (Join-Path $sourceRoot 'pretrain_riceseg\outputs')
  )
  $excludeFiles = @('*.pth','*.pt','*.ckpt','*.onnx','*.bin','*.safetensors','*.npy','*.npz','*.pyc')

  Write-Host '[STEP] Running robocopy sync'
  & robocopy $sourceRoot $tempRepo /E /R:1 /W:1 /XD $excludeDirs /XF $excludeFiles | Out-Host
  if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
  }

  Get-ChildItem -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Get-ChildItem -Recurse -File -Include *.pyc -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

  $gitignoreLines = @(
    '# Python',
    '__pycache__/',
    '*.py[cod]',
    '*.pyo',
    '',
    '# Virtual envs',
    '.venv/',
    'venv/',
    'rdd-dev/',
    '',
    '# Local temp/cache',
    '.tmp/',
    '.repo_publish_tmp/',
    '.repo_tag_tmp/',
    '*.log',
    '',
    '# Data and experiment artifacts',
    'data/',
    'outputs/',
    'visual/',
    'pretrain_riceseg/outputs/',
    '',
    '# Model weights / checkpoints',
    '*.pth',
    '*.pt',
    '*.ckpt',
    '*.onnx',
    '*.bin',
    '*.safetensors',
    '',
    '# Numpy data artifacts',
    '*.npy',
    '*.npz',
    '',
    '# OS/editor',
    '.DS_Store',
    'Thumbs.db',
    '.vscode/',
    '.idea/'
  )
  $gitignoreWritten = $false
  for ($i = 0; $i -lt 8 -and -not $gitignoreWritten; $i++) {
    try {
      $gitignoreLines | Set-Content -Path '.gitignore' -Encoding UTF8 -ErrorAction Stop
      $gitignoreWritten = $true
    } catch {
      Start-Sleep -Milliseconds 300
    }
  }
  if (-not $gitignoreWritten) {
    Write-Host '[WARN] Could not overwrite .gitignore due to file lock; keep existing .gitignore.'
  }

  git add -A
  $status = git status --porcelain
  if (-not [string]::IsNullOrWhiteSpace(($status | Out-String))) {
    git commit -m $commitMessage | Out-Host
    Invoke-GitNoProxy "push origin $branch"
  } else {
    Write-Host '[INFO] No code changes detected; skip commit/push branch.'
  }

  $newHead = (git rev-parse HEAD).Trim()
  Write-Host "[INFO] New HEAD (A7-s2 tag target): $newHead"

  $a7RemoteExists = -not [string]::IsNullOrWhiteSpace((Invoke-GitNoProxy "ls-remote --tags origin refs/tags/$tagA7" | Out-String))
  $a7TaggedNow = $false
  if (-not $a7RemoteExists) {
    git tag -a $tagA7 $newHead -m "A7-s2 fixed-baseline snapshot after blind decode acceptance (2026-03-18)"
    $a7TaggedNow = $true
    Write-Host "[INFO] Created tag: $tagA7 -> $newHead"
  } else {
    Write-Host "[INFO] Remote tag already exists, skip create: $tagA7"
  }

  if ($a4TaggedNow) {
    Invoke-GitNoProxy "push origin refs/tags/$tagA4"
  }
  if ($a7TaggedNow) {
    Invoke-GitNoProxy "push origin refs/tags/$tagA7"
  }

  Write-Host "[DONE] branch=$branch oldHead=$oldHead newHead=$newHead tagA4=$tagA4 tagA7=$tagA7"
}
finally {
  Pop-Location
}

if (Test-Path $tempRepo) {
  for ($i = 0; $i -lt 8; $i++) {
    try {
      Remove-Item -Recurse -Force $tempRepo -ErrorAction Stop
      break
    } catch {
      Start-Sleep -Milliseconds 300
      if ($i -eq 7) { throw }
    }
  }
}
