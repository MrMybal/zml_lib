param(
    [string]$Fork = (Join-Path $PSScriptRoot '..\..\..\zml_win'),
    [string]$BinDir = $env:ZML_BIN_DIR,
    [string]$ForkBinDir = $env:ZML_FORK_BIN_DIR,
    [string]$Logs = (Join-Path $PSScriptRoot '..\..\validation-output\windows-x64'),
    [string]$PeTool,
    [switch]$ForkOnly,
    [switch]$ConsumerOnly
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Fork = (Resolve-Path -LiteralPath $Fork).Path
$work = Join-Path $Fork 'bazel-win-tools'
$Logs = [IO.Path]::GetFullPath($Logs)
[void](New-Item -ItemType Directory -Force $Logs)
if ($ForkOnly -and $ConsumerOnly) { throw 'Use one of -ForkOnly or -ConsumerOnly' }
if (!$BinDir) { $BinDir = Join-Path $root 'validation-output\bin' }
if (!$ForkBinDir) { throw 'Use build.py --with-fork -- --then, or pass -ForkBinDir with the MNIST runfiles' }
$ForkBinDir = (Resolve-Path -LiteralPath $ForkBinDir).Path
$pin = (Get-Content (Join-Path $root 'zml.lock.json') -Raw | ConvertFrom-Json).windows_revision
$actual = & git -C $Fork rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve the zml_win commit' }
if ($actual -ne $pin) {
    & git -C $Fork merge-base --is-ancestor $pin HEAD
    if ($LASTEXITCODE -ne 0) { throw "Expected a descendant of zml_win $pin, got $actual" }
    & git -C $Fork diff --quiet $pin HEAD -- . ':(exclude)README.md' ':(exclude)docs/**'
    if ($LASTEXITCODE -ne 0) { throw "zml_win sources differ from $pin at $actual" }
}
$dirty = & git -C $Fork status --porcelain --untracked-files=no
if ($LASTEXITCODE -ne 0 -or $dirty) { throw 'The tracked files in zml_win must match the pinned commit' }
$portRevision = & git -C $root rev-parse --verify HEAD 2>$null
if ($LASTEXITCODE -ne 0) { $portRevision = '(working tree before first commit)' }
"zml_win $actual (source pin $pin)", "zml_lib $portRevision", (& git -C $root status --short --branch) |
    Set-Content (Join-Path $Logs 'repositories.log')
$mnistDir = Join-Path $ForkBinDir 'examples\mnist'
$assets = @{}
foreach ($line in Get-Content (Join-Path $mnistDir 'mnist.exe.runfiles_manifest')) {
    if ($line -match '^\S+/((?:mnist\.safetensors)|(?:t10k-images\.idx3-ubyte)) (.+)$') {
        $assets[$Matches[1]] = (Resolve-Path -LiteralPath $Matches[2]).Path
    }
}
$weights = $assets['mnist.safetensors']
$images = $assets['t10k-images.idx3-ubyte']
if (!$weights -or !$images) { throw 'MNIST model or test images missing from Bazel runfiles' }
$env:IREE_PJRT_COMPILER_LIB_PATH = Join-Path $work 'iree-wheel\iree\compiler\_mlir_libs\IREECompiler.dll'
$env:HIP_VISIBLE_DEVICES = '0'
$env:IREE_PJRT_ROCM_TARGET = 'gfx1036'

$hardware = @(
    @{Backend='cpu'; Pjrt='cpu'; Device='cpu:0'; Variable='ZML_CPU_PJRT_DLL'; Dll=(Join-Path $work 'pjrt-cpu\pjrt_cpu.dll') },
    @{Backend='cuda'; Pjrt='iree_cuda'; Device='RTX 3090'; Variable='ZML_CUDA_PJRT_DLL'; Dll=(Join-Path $work 'iree-build\python\iree\_pjrt_libs\cuda\iree_pjrt_cuda_dylib.dll') },
    @{Backend='rocm'; Pjrt='iree_rocm'; Device='AMD Radeon'; Variable='ZML_ROCM_PJRT_DLL'; Dll=(Join-Path $work 'iree-build\python\iree\_pjrt_libs\rocm\iree_pjrt_rocm_dylib.dll') },
    @{Backend='vulkan'; Pjrt='iree_vulkan'; Device='AMD Radeon'; Variable='ZML_VULKAN_PJRT_DLL'; Dll=(Join-Path $work 'iree-build\python\iree\_pjrt_libs\vulkan\iree_pjrt_vulkan_dylib.dll') }
)
$amd = Get-ChildItem "$env:SystemRoot\System32\DriverStore\FileRepository" -Filter 'u0420529.inf_amd64_*' -Directory |
    Get-ChildItem -Recurse -Filter amd-vulkan64.json | Select-Object -First 1
if ($amd) { $env:VK_DRIVER_FILES = $amd.FullName }
$failures = [System.Collections.Generic.List[string]]::new()

function Run-Logged($Program, $Arguments, $Label, $Patterns) {
    $file = Join-Path $Logs "$Label.log"
    & $Program @Arguments *> $file
    if ($LASTEXITCODE -ne 0) {
        $failures.Add("$Label exited $LASTEXITCODE")
        return $false
    }
    $output = Get-Content -LiteralPath $file -Raw
    foreach ($pattern in $Patterns) {
        if ($output -notmatch $pattern) {
            $failures.Add("$Label missing expected pattern $pattern")
            return $false
        }
    }
    return $true
}

if (!$ForkOnly) {
    $BinDir = (Resolve-Path -LiteralPath $BinDir).Path
    $consumer = Join-Path $BinDir 'windows_consumer.exe'
    $dll = Join-Path $BinDir 'zml_lib.dll'
    if (!$PeTool) {
        $command = Get-Command llvm-readobj -ErrorAction SilentlyContinue
        if ($command) { $PeTool = $command.Source }
        else { $PeTool = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin\llvm-readobj.exe' }
    }
    $header = Get-Content (Join-Path $root 'include\zml_lib.h') -Raw
    $symbols = [regex]::Matches($header, '\b(zml_lib_[a-z_]+)\(') | ForEach-Object {
        'Name:\s+' + $_.Groups[1].Value + '\s*$'
    }
    [void](Run-Logged $PeTool @('--file-headers','--coff-exports','--coff-imports',$dll) 'consumer-pe-library' (@('IMAGE_FILE_MACHINE_AMD64','IMAGE_FILE_DLL') + ($symbols | ForEach-Object {"(?m)$_"})))
    [void](Run-Logged $PeTool @('--file-headers','--coff-imports',$consumer) 'consumer-pe-executable' @('IMAGE_FILE_MACHINE_AMD64','(?im)Name:\s+zml_lib\.dll'))
    [void](Run-Logged $PeTool @('--file-headers','--coff-imports',$env:IREE_PJRT_COMPILER_LIB_PATH) 'pjrt-pe-iree-compiler' @('IMAGE_FILE_MACHINE_AMD64'))
    Get-FileHash -LiteralPath $dll,$consumer | Format-Table -AutoSize | Out-String |
        Set-Content (Join-Path $Logs 'consumer-sha256.log')
}

$rows = foreach ($entry in $hardware) {
    $backend = $entry.Backend
    [Environment]::SetEnvironmentVariable($entry.Variable, $entry.Dll, 'Process')
    if ($backend -eq 'cuda') { $env:IREE_PJRT_IREE_COMPILER_OPTIONS = '--iree-cuda-target=sm_86' }
    else { Remove-Item Env:IREE_PJRT_IREE_COMPILER_OPTIONS -ErrorAction SilentlyContinue }
    $device = [regex]::Escape($entry.Device)
    $name = [regex]::Escape($entry.Pjrt)
    $direct = $null
    $client = $null
    $unavailable = $null
    if ($backend -eq 'vulkan' -and !$amd) { $failures.Add('AMD Vulkan ICD manifest missing') }
    elseif (!$ConsumerOnly) {
        $smoke = Join-Path $ForkBinDir 'examples\native_smoke\native_smoke.exe'
        $direct = Run-Logged $smoke @($backend) "fork-native-smoke-$backend" @("PJRT client platform: $name", '19 \+ 23 = 42', 'Inference 10: \{ 7, 0 \}', 'Released model executable and buffers')
        $expected = 7,2,1
        foreach ($index in 0..2) {
            $result = Run-Logged (Join-Path $mnistDir 'mnist.exe') @($weights,$images,$backend,$index) "fork-mnist-$backend-$index" @("PJRT: $name, device: .*$device", "MNIST sample $index, predicted digit $($expected[$index])")
            $direct = $result -and $direct
        }
    }
    if (!$ForkOnly) {
        if (Test-Path -LiteralPath $entry.Dll) {
            [void](Run-Logged $PeTool @('--file-headers','--coff-exports','--coff-imports',$entry.Dll) "pjrt-pe-$backend" @('IMAGE_FILE_MACHINE_AMD64','Name:\s+GetPjrtApi'))
        }
        if ($backend -ne 'vulkan' -or $amd) {
            $client = Run-Logged $consumer @($backend,$weights,$images) "consumer-$backend" @("pjrt=$name device=.*$device.* checked=1", '19 \+ 23 = 42', 'smoke=\[7,0\]', 'mnist\[0\]=7', 'mnist\[1\]=2', 'mnist\[2\]=1', 'mnist_reloaded=7')
        }
        try {
            [Environment]::SetEnvironmentVariable($entry.Variable, (Join-Path $Logs 'missing-plugin.dll'), 'Process')
            $unavailable = Run-Logged $consumer @($backend,'--unavailable') "consumer-missing-$backend" @("backend=$backend unavailable status=3 error=\S+")
        } finally {
            [Environment]::SetEnvironmentVariable($entry.Variable, $entry.Dll, 'Process')
        }
    }
    [PSCustomObject]@{Backend=$backend; Pjrt=$entry.Pjrt; Device=$entry.Device; Fork=$direct; Cpp=$client; Missing=$unavailable; Dll=$entry.Dll}
}
$rows | Export-Csv -NoTypeInformation (Join-Path $Logs 'matrix.csv')
$rows | Format-Table -Property Backend,Pjrt,Device,Fork,Cpp,Missing
if ($failures.Count) { throw ($failures -join [Environment]::NewLine) }
