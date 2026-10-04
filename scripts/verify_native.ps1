# MSVC qualification supplements the g++/clang++ pytest harness on Windows.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskOut = Join-Path $taskRoot '.local\native-validation'
New-Item -ItemType Directory -Force -Path $taskOut | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'MSVC C++ tools unavailable. Native behavior was not tested.' }
$devCommand = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
$binary = Join-Path $taskOut 'scheduler.exe'
$nativeInclude = Join-Path $taskRoot 'native'
$source = Join-Path $taskRoot 'tests\native_input_scheduler.cpp'
$object = Join-Path $taskOut 'scheduler.obj'
$compile = 'call "' + $devCommand + '" -arch=x64 -host_arch=x64 >nul && cl /nologo /std:c++20 /EHsc /W4 /WX /UNDEBUG /I"' + $nativeInclude + '" "' + $source + '" /Fe"' + $binary + '" /Fo"' + $object + '"'
& $env:ComSpec /d /s /c $compile
if ($LASTEXITCODE -ne 0) { throw 'Native harness compilation failed.' }
$cases = [regex]::Matches((Get-Content -Raw -Encoding UTF8 $source), 'CASE\((\w+)\)') | ForEach-Object { $_.Groups[1].Value } | Where-Object { $_ -ne 'n' } | Select-Object -Unique
foreach ($case in $cases) {
    & $binary $case
    if ($LASTEXITCODE -ne 0) { throw "Native case failed: $case" }
}
Write-Output ("Native MSVC validation: " + $cases.Count + ' cases passed; assertions enabled.')
