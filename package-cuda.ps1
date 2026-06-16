# package-cuda.ps1 — Create Qwen3-TTS portable ZIP with CUDA support
# Packs CUDA executables + CUDA DLLs + default model (1.7B-CustomVoice) + shared components.

$ErrorActionPreference = "Stop"
$root = "D:\qwen3-tts-rscandle"
$exeCli = "$root\target\release\generate_audio.exe"
$exeGui = "$root\target\release\gui.exe"
$modelVariant = "$root\models\1.7B-CustomVoice"
$modelSt  = "$root\models\speech_tokenizer"
$modelTok = "$root\models\tokenizer"

$cudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"
$vsRedist = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC\14.51.36231\x64\Microsoft.VC145.CRT"

# CUDA DLLs needed (from dumpbin /dependents analysis)
$cudaDlls = @{
    "cublas64_13.dll"   = "$cudaBin\cublas64_13.dll"
    "cublasLt64_13.dll" = "$cudaBin\cublasLt64_13.dll"
    "curand64_10.dll"   = "$cudaBin\curand64_10.dll"
}

$vcruntimeDlls = @{
    "vcruntime140.dll"   = "$vsRedist\vcruntime140.dll"
    "vcruntime140_1.dll" = "$vsRedist\vcruntime140_1.dll"
}

# Output filename with CUDA version tag
$cudaVer = "13.2"
$out = "$root\Qwen3-TTS-v0.4.0-cuda$cudaVer-win64.zip"

# --- Verify all files exist ---
$missing = @()
if (-not (Test-Path $exeCli)) { $missing += "generate_audio.exe" }
if (-not (Test-Path $exeGui)) { $missing += "gui.exe" }
if (-not (Test-Path $modelVariant)) { $missing += "models/1.7B-CustomVoice/" }
if (-not (Test-Path $modelSt))      { $missing += "models/speech_tokenizer/" }
if (-not (Test-Path $modelTok))     { $missing += "models/tokenizer/" }
foreach ($kv in $cudaDlls.GetEnumerator()) {
    if (-not (Test-Path $kv.Value)) { $missing += "CUDA: $($kv.Key)" }
}
foreach ($kv in $vcruntimeDlls.GetEnumerator()) {
    if (-not (Test-Path $kv.Value)) { $missing += $kv.Key }
}
if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing files:`n  $($missing -join "`n  ")" -ForegroundColor Red
    exit 1
}

# File counts
$cudaTotalSize = 0
foreach ($kv in $cudaDlls.GetEnumerator()) {
    $cudaTotalSize += (Get-Item $kv.Value).Length
}
Write-Host "CUDA DLLs total: $( '{0:N1}' -f ($cudaTotalSize / 1MB) ) MB" -ForegroundColor Yellow

# Remove old zip
if (Test-Path $out) { Remove-Item -LiteralPath $out -Force }

Write-Host "Creating $out ..." -ForegroundColor Cyan

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($out, [System.IO.Compression.ZipArchiveMode]::Create)

function Add-FileToZip($zip, $sourcePath, $entryName) {
    $entry = $zip.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Fastest)
    $entryStream = $entry.Open()
    $fileStream = [System.IO.File]::OpenRead($sourcePath)
    try {
        $fileStream.CopyTo($entryStream)
    }
    finally {
        $fileStream.Close()
        $entryStream.Close()
    }
}

function Add-DirectoryToZip($zip, $sourceDir, $zipPrefix) {
    $files = Get-ChildItem -LiteralPath $sourceDir -Recurse -File
    $total = $files.Count
    $i = 0
    foreach ($f in $files) {
        $i++
        $relPath = "$zipPrefix/$($f.FullName.Substring($sourceDir.Length + 1))" -replace '\\', '/'
        $pct = [math]::Round($i / $total * 100)
        Write-Progress -Activity "Packing $zipPrefix" -Status "$relPath" -PercentComplete $pct -CurrentOperation "$i / $total"
        Add-FileToZip $zip $f.FullName $relPath
    }
    Write-Progress -Activity "Packing $zipPrefix" -Completed
}

try {
    # --- generate_audio.exe (CLI) ---
    Write-Host "  Adding generate_audio.exe"
    Add-FileToZip $zip $exeCli "generate_audio.exe"

    # --- gui.exe (GUI) ---
    Write-Host "  Adding gui.exe"
    Add-FileToZip $zip $exeGui "gui.exe"

    # --- CUDA DLLs ---
    foreach ($kv in $cudaDlls.GetEnumerator()) {
        Write-Host "  Adding CUDA: $($kv.Key)"
        Add-FileToZip $zip $kv.Value $kv.Key
    }

    # --- VCRUNTIME DLLs ---
    foreach ($kv in $vcruntimeDlls.GetEnumerator()) {
        Write-Host "  Adding $($kv.Key)"
        Add-FileToZip $zip $kv.Value $kv.Key
    }

    # --- models/1.7B-CustomVoice/ (default model variant) ---
    Write-Host "  Packing models/1.7B-CustomVoice/ ..."
    Add-DirectoryToZip $zip $modelVariant "models/1.7B-CustomVoice"

    # --- models/speech_tokenizer/ (shared) ---
    Write-Host "  Packing models/speech_tokenizer/ ..."
    Add-DirectoryToZip $zip $modelSt "models/speech_tokenizer"

    # --- models/tokenizer/ (shared) ---
    Write-Host "  Packing models/tokenizer/ ..."
    Add-DirectoryToZip $zip $modelTok "models/tokenizer"

    $finalSize = (Get-Item $out).Length
    Write-Host "Done! $( '{0:N1}' -f ($finalSize / 1GB) ) GB" -ForegroundColor Green
}
finally {
    $zip.Dispose()
}
