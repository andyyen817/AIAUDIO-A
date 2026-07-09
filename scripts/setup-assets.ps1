param(
    [switch]$Download
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Downloads = Join-Path $Root "downloads"
$AppLibRoot = Join-Path $Root "app\src\main\jniLibs"
$ModelRoot = Join-Path $Root "app\src\main\assets\sherpa-onnx-paraformer-zh-small-2024-03-09"

$AndroidLibUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.3/sherpa-onnx-v1.13.3-android.tar.bz2"
$ModelUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2"

New-Item -ItemType Directory -Force -Path $Downloads, $AppLibRoot, $ModelRoot | Out-Null

if ($Download) {
    $androidTar = Join-Path $Downloads "sherpa-onnx-v1.13.3-android.tar.bz2"
    $modelTar = Join-Path $Downloads "sherpa-onnx-paraformer-zh-small-2024-03-09.tar.bz2"

    if (!(Test-Path $androidTar)) {
        curl.exe -L --retry 3 --connect-timeout 20 -o $androidTar $AndroidLibUrl
    }
    if (!(Test-Path $modelTar)) {
        curl.exe -L --retry 3 --connect-timeout 20 -o $modelTar $ModelUrl
    }

    $extractDir = Join-Path $Downloads "extract"
    New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

    tar.exe -xjf $androidTar -C $extractDir
    foreach ($abi in @("arm64-v8a", "x86_64")) {
        $src = Join-Path $extractDir "jniLibs\$abi"
        $dst = Join-Path $AppLibRoot $abi
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item -Path (Join-Path $src "*") -Destination $dst -Force
    }

    tar.exe -xjf $modelTar -C $extractDir
    $modelExtract = Join-Path $extractDir "sherpa-onnx-paraformer-zh-small-2024-03-09"
    Copy-Item -Path (Join-Path $modelExtract "model.int8.onnx") -Destination $ModelRoot -Force
    Copy-Item -Path (Join-Path $modelExtract "tokens.txt") -Destination $ModelRoot -Force
}

Write-Host "Expected JNI library folders:"
foreach ($abi in @("arm64-v8a", "x86_64")) {
    $so = Join-Path $AppLibRoot "$abi\libsherpa-onnx-jni.so"
    Write-Host ("- {0}: {1}" -f $abi, $(if (Test-Path $so) { "OK" } else { "missing" }))
}

Write-Host "Expected model files:"
foreach ($file in @("model.int8.onnx", "tokens.txt")) {
    $path = Join-Path $ModelRoot $file
    Write-Host ("- {0}: {1}" -f $file, $(if (Test-Path $path) { "OK" } else { "missing" }))
}
