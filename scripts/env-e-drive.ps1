$env:ANDROID_HOME = "E:\Android\Sdk"
$env:ANDROID_SDK_ROOT = "E:\Android\Sdk"
$env:GRADLE_USER_HOME = "E:\Android\.gradle"
$env:ANDROID_AVD_HOME = "E:\Android\.android\avd"
$env:PATH = "$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator;$env:ANDROID_HOME\cmdline-tools\latest\bin;$env:PATH"

Write-Host "ANDROID_HOME=$env:ANDROID_HOME"
Write-Host "GRADLE_USER_HOME=$env:GRADLE_USER_HOME"
Write-Host "ANDROID_AVD_HOME=$env:ANDROID_AVD_HOME"

