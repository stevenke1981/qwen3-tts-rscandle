# package.ps1 — Create Qwen3-TTS portable ZIP (CPU version)
# Packs executables + default model (1.7B-CustomVoice) + shared components.

$ErrorActionPreference = "Stop"
$root = "D:\qwen3-tts-rscandle"
$exeCli = "$root\target\release\generate_audio.exe"
$exeGui = "$root\target\release\gui.exe"
$modelVariant = "$root\models\1.7B-CustomVoice"
$modelSt  = "$root\models\speech_tokenizer"
$modelTok = "$root\models\tokenizer"
$out  = "$root\Qwen3-TTS-v0.4.0-win64.zip"
$vcruntime  = "C:\Windows\System32\vcruntime140.dll"
$vcruntime1 = "C:\Windows\System32\vcruntime140_1.dll"

# Verify files exist
$missing = @()
if (-not (Test-Path $exeCli)) { $missing += "generate_audio.exe" }
if (-not (Test-Path $exeGui)) { $missing += "gui.exe" }
if (-not (Test-Path $modelVariant)) { $missing += "models/1.7B-CustomVoice/" }
if (-not (Test-Path $modelSt))      { $missing += "models/speech_tokenizer/" }
if (-not (Test-Path $modelTok))     { $missing += "models/tokenizer/" }
if (-not (Test-Path $vcruntime))  { $missing += "vcruntime140.dll" }
if (-not (Test-Path $vcruntime1)) { $missing += "vcruntime140_1.dll" }
if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing files:`n  $($missing -join "`n  ")" -ForegroundColor Red
    exit 1
}

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

    # --- vcruntime DLLs ---
    foreach ($dll in @($vcruntime, $vcruntime1)) {
        $name = Split-Path -Leaf $dll
        Write-Host "  Adding $name"
        Add-FileToZip $zip $dll $name
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
