#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Download all Qwen3-TTS model variants to the shared models/ directory.
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ModelsDir = "$Root\models"

$SharedDir   = "$ModelsDir\shared"
$TokenizerDir  = "$SharedDir\tokenizer"
$SpTokenizerDir = "$SharedDir\speech_tokenizer"

# Variant definitions
$VARIANT_HF = [ordered]@{
    "0.6B-Base" = @{ repo = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";          size = "1.8 GB" }
    "0.6B-CustomVoice" = @{ repo = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"; size = "1.8 GB" }
    "1.7B-CustomVoice" = @{ repo = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"; size = "3.9 GB" }
    "1.7B-VoiceDesign" = @{ repo = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"; size = "3.8 GB" }
}

$REQUIRED_VARIANT_FILES = @("model.safetensors", "config.json")

$SHARED_DOWNLOADS = @(
    @{ name = "Speech tokenizer: model.safetensors"; dest = "$SpTokenizerDir\model.safetensors"; url = "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/model.safetensors" },
    @{ name = "Speech tokenizer: config.json";       dest = "$SpTokenizerDir\config.json";       url = "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/config.json" },
    @{ name = "Speech tokenizer: preprocessor_config.json"; dest = "$SpTokenizerDir\preprocessor_config.json"; url = "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/preprocessor_config.json" },
    @{ name = "Speech tokenizer: configuration.json"; dest = "$SpTokenizerDir\configuration.json"; url = "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/configuration.json" },
    @{ name = "Text tokenizer: tokenizer.json (Qwen2-0.5B)"; dest = "$TokenizerDir\tokenizer.json"; url = "https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/tokenizer.json" }
)

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Download-File([string]$url, [string]$dest, [string]$label) {
    if (Test-Path $dest) {
        $size = (Get-Item $dest).Length
        Write-Host "  [SKIP] $label -- already exists ($( '{0:N1}' -f ($size / 1MB) ) MB)" -ForegroundColor Yellow
        return $true
    }

    $parent = Split-Path -Parent $dest
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }

    Write-Host "  Downloading $label ..." -ForegroundColor Green
    Write-Host "    $url"
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($url, $dest)
        $size = (Get-Item $dest).Length
        Write-Host "    -> $( '{0:N1}' -f ($size / 1MB) ) MB" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "    FAILED: $_" -ForegroundColor Red
        if (Test-Path $dest) { Remove-Item $dest -Force }
        return $false
    }
}

# Main
Write-Host ""
Write-Host "========================================" -ForegroundColor Magenta
Write-Host "  Qwen3-TTS Model Downloader v0.4.0" -ForegroundColor Magenta
Write-Host "  Models directory: $ModelsDir" -ForegroundColor Magenta
Write-Host "========================================" -ForegroundColor Magenta

# Step 1: Shared files
Write-Step "Shared files (speech_tokenizer + text tokenizer)"

$allSharedOk = $true
foreach ($dl in $SHARED_DOWNLOADS) {
    if (-not (Download-File $dl.url $dl.dest $dl.name)) {
        $allSharedOk = $false
    }
}

# Step 2: Variant model weights
Write-Step "Model variants"

# 1.7B-Base check
if (Test-Path "$ModelsDir\1.7B-Base\model.safetensors") {
    $size = (Get-Item "$ModelsDir\1.7B-Base\model.safetensors").Length
    Write-Host "  [OK] 1.7B-Base -- $( '{0:N1}' -f ($size / 1GB) ) GB (from existing test_data/model)" -ForegroundColor Yellow
}

$successCount = 0
$skipCount = 0
$failCount = 0

foreach ($variant in $VARIANT_HF.Keys) {
    $info = $VARIANT_HF[$variant]
    $variantDir = "$ModelsDir\$variant"
    if (-not (Test-Path $variantDir)) { New-Item -ItemType Directory -Path $variantDir -Force | Out-Null }

    $alreadyComplete = $true
    foreach ($file in $REQUIRED_VARIANT_FILES) {
        $fp = "$variantDir\$file"
        if (-not (Test-Path $fp) -or (Get-Item $fp).Length -eq 0) {
            $alreadyComplete = $false
            break
        }
    }

    if ($alreadyComplete) {
        Write-Host "  [SKIP] $variant -- already downloaded" -ForegroundColor Yellow
        $skipCount++
        continue
    }

    Write-Host ""
    Write-Host "-- $variant ($($info.size)) --" -ForegroundColor Green

    $variantOk = $true
    foreach ($file in $REQUIRED_VARIANT_FILES) {
        $url = "https://huggingface.co/$($info.repo)/resolve/main/$file"
        $dest = "$variantDir\$file"
        if (-not (Download-File $url $dest "$variant/$file")) {
            $variantOk = $false
        }
    }

    if ($variantOk) {
        Write-Host "  DONE: $variant complete!" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "  FAILED: $variant" -ForegroundColor Red
        $failCount++
    }
}

# Summary
Write-Step "Download Summary"

if ($allSharedOk) {
    Write-Host "  Shared files: OK" -ForegroundColor Green
} else {
    Write-Host "  Shared files: PARTIAL" -ForegroundColor Yellow
}

Write-Host "  Variants:"
Write-Host "    - 1.7B-Base:       from test_data/model" -ForegroundColor Yellow
$idx = 0
foreach ($variant in $VARIANT_HF.Keys) {
    if ($successCount -gt $idx) {
        Write-Host "    - ${variant}: DOWNLOADED" -ForegroundColor Green
    } elseif ($skipCount -gt $idx) {
        Write-Host "    - ${variant}: SKIPPED (already exists)" -ForegroundColor Yellow
    } else {
        Write-Host "    - ${variant}: PENDING" -ForegroundColor Yellow
    }
    $idx++
}

Write-Host ""
Write-Host "  Usage examples:" -ForegroundColor Cyan
Write-Host "    generate_audio --model-dir models/1.7B-CustomVoice --tokenizer-dir models/shared/tokenizer --speaker vivian --text hello"
Write-Host "    gui.exe"
Write-Host ""
Write-Host "Done!" -ForegroundColor Magenta
