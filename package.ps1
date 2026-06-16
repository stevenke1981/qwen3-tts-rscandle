# package.ps1 — Create Qwen3-TTS portable ZIP

$root = "D:\qwen3-tts-rscandle"
$exe  = "$root\target\release\gui.exe"
$model = "$root\test_data\model"
$out  = "$root\Qwen3-TTS-v0.3.0-win64.zip"
$vcruntime  = "C:\Windows\System32\vcruntime140.dll"
$vcruntime1 = "C:\Windows\System32\vcruntime140_1.dll"

# Verify files exist
$missing = @()
if (-not (Test-Path $exe))   { $missing += "gui.exe" }
if (-not (Test-Path $model)) { $missing += "model/" }
if (-not (Test-Path $vcruntime))  { $missing += "vcruntime140.dll" }
if (-not (Test-Path $vcruntime1)) { $missing += "vcruntime140_1.dll" }
if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing files: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}

# Remove old zip
if (Test-Path $out) { Remove-Item -LiteralPath $out -Force }

Write-Host "Creating $out ..." -ForegroundColor Cyan

Add-Type -AssemblyName System.IO.Compression
$zip = [System.IO.Compression.ZipFile]::Open($out, [System.IO.Compression.ZipArchiveMode]::Create)

function Add-FileToZip($zip, $sourcePath, $entryName) {
    $entry = $zip.CreateEntry($entryName, [System.IO.Compression.CompressionLevel]::Optimal)
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

try {
    # --- gui.exe ---
    Write-Host "  Adding gui.exe"
    Add-FileToZip $zip $exe "gui.exe"

    # --- vcruntime DLLs ---
    foreach ($dll in @($vcruntime, $vcruntime1)) {
        $name = Split-Path -Leaf $dll
        Write-Host "  Adding $name"
        Add-FileToZip $zip $dll $name
    }

    # --- model/ directory ---
    $modelFiles = Get-ChildItem -LiteralPath $model -Recurse -File
    $total = $modelFiles.Count
    $i = 0
    foreach ($f in $modelFiles) {
        $i++
        $relPath = "model/$($f.FullName.Substring($model.Length + 1))" -replace '\\', '/'
        $pct = [math]::Round($i / $total * 100)
        Write-Progress -Activity "Packing model files" -Status "$relPath" -PercentComplete $pct -CurrentOperation "$i / $total"
        Add-FileToZip $zip $f.FullName $relPath
    }
    Write-Progress -Activity "Packing model files" -Completed

    $finalSize = (Get-Item $out).Length
    Write-Host "Done! $( '{0:N1}' -f ($finalSize / 1GB) ) GB" -ForegroundColor Green
}
finally {
    $zip.Dispose()
}
