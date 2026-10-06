# MSVC qualification supplements the g++/clang++ pytest harness on Windows.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$taskOut = Join-Path $taskRoot '.local\native-validation'
New-Item -ItemType Directory -Force -Path $taskOut | Out-Null
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $installation) { throw 'MSVC C++ tools unavailable. Native behavior was not tested.' }
$devCommand = Join-Path $installation 'Common7\Tools\VsDevCmd.bat'
$nativeInclude = Join-Path $taskRoot 'native'
$harnesses = @(
    @{ Name = 'scheduler'; Source = 'native_input_scheduler.cpp' },
    @{ Name = 'walking-clearance'; Source = 'native_walking_clearance.cpp' },
    @{ Name = 'navigation-resolution'; Source = 'native_navigation_resolution.cpp' },
    @{ Name = 'navigation-refinement'; Source = 'native_navigation_refinement.cpp' },
    @{ Name = 'container-pose'; Source = 'native_container_pose.cpp' },
    @{ Name = 'dialogue-observation'; Source = 'native_dialogue_observation.cpp' },
    @{ Name = 'progress-autosave'; Source = 'native_progress_autosave.cpp' }
)
foreach ($harness in $harnesses) {
    $binary = Join-Path $taskOut ($harness.Name + '.exe')
    $source = Join-Path $taskRoot ('tests\' + $harness.Source)
    $object = Join-Path $taskOut ($harness.Name + '.obj')
    $compile = 'call "' + $devCommand + '" -arch=x64 -host_arch=x64 >nul && cl /nologo /std:c++20 /EHsc /W4 /WX /UNDEBUG /I"' + $nativeInclude + '" "' + $source + '" /Fe"' + $binary + '" /Fo"' + $object + '"'
    & $env:ComSpec /d /s /c $compile
    if ($LASTEXITCODE -ne 0) { throw "Native harness compilation failed: $($harness.Name)" }
    $cases = [regex]::Matches((Get-Content -Raw -Encoding UTF8 $source), 'CASE\((\w+)\)') | ForEach-Object { $_.Groups[1].Value } | Where-Object { $_ -ne 'n' } | Select-Object -Unique
    foreach ($case in $cases) {
        & $binary $case
        if ($LASTEXITCODE -ne 0) { throw "Native case failed: $($harness.Name)/$case" }
    }
    Write-Output ("Native MSVC validation ($($harness.Name)): " + $cases.Count + ' cases passed; checks enabled.')
}
