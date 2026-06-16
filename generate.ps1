#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Run Qwen3-TTS inference on CUDA with timing.
.DESCRIPTION
    Wraps generate_audio.exe with CUDA environment setup and timing.
    If no text given, reads all .txt files from the working directory.
.PARAMETER Text
    Text to synthesize. If omitted, processes all *.txt files.
.PARAMETER Output
    Output WAV path (single mode only).
.PARAMETER Seed
    Random seed (default: 42).
.PARAMETER Duration
    Max duration in seconds (overrides --frames).
.PARAMETER Frames
    Max frames (default: 2048, ~164s).
.PARAMETER Speaker
    Preset speaker: ryan, serena, vivian, aiden, etc. (default: ryan).
.PARAMETER Language
    Language: english, chinese, japanese (default: english).
.PARAMETER Device
    Device override (default: auto → CUDA if available).
.PARAMETER ModelDir
    Model directory (default: test_data/model).
.PARAMETER OutputDir
    Output directory for batch mode (default: test_data/rust_audio).
.PARAMETER NoBuild
    Skip cargo build check (use existing binary).
.PARAMETER Release
    Use release binary instead of debug.
.PARAMETER Temperature
    Sampling temperature (default: 0.7).
.PARAMETER TopK
    Top-k sampling (default: 50).
.PARAMETER TopP
    Top-p sampling (default: 0.9).
.PARAMETER RepetitionPenalty
    Repetition penalty (default: 1.05).
.PARAMETER Quiet
    Suppress timing output.
.EXAMPLE
    .\generate.ps1 -Text "Hello, this is a test."
.EXAMPLE
    .\generate.ps1 -Text "你好世界" -Language chinese -Speaker vivian -Duration 10
.EXAMPLE
    .\generate.ps1 -Duration 30 -Output demo.wav
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Text,

    [string]$Output,
    [int]$Seed = 42,
    [double]$Duration,
    [int]$Frames = 2048,
    [string]$Speaker = "ryan",
    [string]$Language = "english",
    [string]$Device = "auto",
    [string]$ModelDir = "test_data/model",
    [string]$OutputDir = "output/voice",
    [switch]$NoBuild,
    [switch]$Release,
    [double]$Temperature = 0.7,
    [int]$TopK = 50,
    [double]$TopP = 0.9,
    [double]$RepetitionPenalty = 1.05,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Binary = if ($Release) { "target/release/generate_audio.exe" } else { "target/debug/generate_audio.exe" }
$BinaryPath = Join-Path $ProjectRoot $Binary
$ModelDirPath = Join-Path $ProjectRoot $ModelDir
$OutputDirPath = Join-Path $ProjectRoot $OutputDir

# ── 1. Ensure CUDA_PATH points to v13.2 ──────────────────────────────────
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if (Test-Path "$cudaRoot\bin\nvcc.exe") {
    $env:CUDA_PATH = $cudaRoot
    $env:CUDA_TOOLKIT_ROOT_DIR = $cudaRoot
}

# ── 2. Build if needed ────────────────────────────────────────────────────
if (-not $NoBuild -and -not (Test-Path $BinaryPath)) {
    Write-Host ">>> Binary not found at $Binary — building..." -ForegroundColor Yellow
    $buildArgs = @("build", "--features", "cli,cuda")
    if ($Release) { $buildArgs += "--release" }

    $proc = Start-Process -FilePath "cargo" -ArgumentList $buildArgs -NoNewWindow -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        Write-Error "Cargo build failed (exit $($proc.ExitCode))"
        exit 1
    }
    Write-Host "<<< Build complete." -ForegroundColor Green
}

if (-not (Test-Path $BinaryPath)) {
    Write-Error "Binary not found at $BinaryPath — run build-cuda.bat first or use -NoBuild"
    exit 1
}

# ── 3. Ensure model directory exists ──────────────────────────────────────
if (-not (Test-Path "$ModelDirPath\model.safetensors")) {
    Write-Warning "Model not found at $ModelDirPath — run .\scripts\download_test_data.sh first"
}

# ── 4. Collect texts to process ──────────────────────────────────────────
$texts = @()
if ($Text) {
    $texts += @{ Text = $Text; Label = $Text.Substring(0, [Math]::Min(40, $Text.Length)) }
}
else {
    $txtFiles = Get-ChildItem -Path $ProjectRoot -Filter "*.txt" | Where-Object { $_.Name -ne "Cargo.lock" }
    if ($txtFiles.Count -eq 0) {
        Write-Error "No -Text given and no *.txt files found in project root. Provide -Text or create a .txt file."
        exit 1
    }
    foreach ($f in $txtFiles) {
        $content = Get-Content $f.FullName -Raw | ForEach-Object { $_.Trim() }
        if ($content) {
            $texts += @{ Text = $content; Label = $f.BaseName }
        }
    }
}

# ── 5. Run inference ─────────────────────────────────────────────────────
$totalStart = [System.Diagnostics.Stopwatch]::StartNew()

for ($i = 0; $i -lt $texts.Count; $i++) {
    $entry = $texts[$i]
    $runLabel = if ($texts.Count -gt 1) { "[$($i+1)/$($texts.Count)] $($entry.Label)" } else { $entry.Label }

    Write-Host ""
    Write-Host ("====== {0} ======" -f $runLabel) -ForegroundColor Cyan

    # Build args
    $argsList = @(
        "--text", "`"$($entry.Text)`""
        "--seed", $Seed
        "--frames", $Frames
        "--temperature", $Temperature
        "--top-k", $TopK
        "--top-p", $TopP
        "--repetition-penalty", $RepetitionPenalty
        "--speaker", $Speaker
        "--language", $Language
        "--device", $Device
        "--model-dir", "`"$ModelDir`""
        "--output-dir", "`"$OutputDir`""
    )

    if ($Duration) {
        # --duration overrides --frames
        $argsList = $argsList | Where-Object { $_ -ne "--frames" -and $_ -ne $Frames }
        $argsList += "--duration", $Duration
    }

    # Determine output path
    if ($Output -and $texts.Count -eq 1) {
        # Explicit -Output provided
        $outPath = $Output
    } else {
        # Auto-generate timestamped filename: output/voice/voice_20260616_025418.wav
        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        $suffix = if ($texts.Count -gt 1) { "_{0}" -f $i } else { "" }
        $outName = "voice_{0}{1}.wav" -f $ts, $suffix
        $outPath = Join-Path $OutputDir $outName
    }
    $argsList += "--output", "`"$outPath`""

    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    # Run the binary. Use Start-Process to get output in real time + capture
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $BinaryPath
    $psi.Arguments = $argsList -join " "
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.WorkingDirectory = $ProjectRoot
    $p = [System.Diagnostics.Process]::Start($psi)
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    $timer.Stop()

    # Print output
    if ($stdout) { Write-Host $stdout -NoNewline }
    if ($stderr) { Write-Host $stderr -ForegroundColor DarkYellow -NoNewline }

    if ($p.ExitCode -eq 0) {
        if (-not $Quiet) {
            Write-Host ""
            Write-Host ("[OK] {0} - {1:0.00}s" -f $runLabel, $timer.Elapsed.TotalSeconds) -ForegroundColor Green
        }
    }
    else {
        Write-Host "[FAIL] $($runLabel) (exit $($p.ExitCode))" -ForegroundColor Red
    }
}

$totalStart.Stop()
Write-Host ""
if ($texts.Count -gt 1 -and -not $Quiet) {
    Write-Host ("====== Total: {0} texts ======" -f $texts.Count) -ForegroundColor Cyan
    Write-Host ("Total: {0} texts in {1:0.00}s" -f $texts.Count, $totalStart.Elapsed.TotalSeconds) -ForegroundColor Cyan
}
