param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_3.exe"
$csvPath = Join-Path $resultsDir "MPI_3.csv"

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

Write-Host "Configuring and building via CMake..."
& cmake -S $ProjectRoot -B $buildDir -DCMAKE_BUILD_TYPE=Release
& cmake --build $buildDir --config Release

$exePath = Join-Path $binDir $exeName
if (-not (Test-Path $exePath)) {
    Write-Error "Executable not found at $exePath. Проверьте сборку."
    exit 1
}

$metadataPath = Join-Path $resultsDir "metadata.txt"
if (-not (Test-Path $metadataPath)) {
    $cmakeInfo = & cmake --version | Select-Object -First 1
    "$cmakeInfo" | Out-File -FilePath $metadataPath -Encoding utf8
    "buildDate: $(Get-Date -Format o)" | Out-File -FilePath $metadataPath -Append -Encoding utf8
}

"testType,messageSizeBytes,numProcesses,numIterations,totalTimeSeconds,avgRoundTripSeconds,bandwidthBytesPerSec,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$messageSizeList = @(1, 8, 64, 512, 4096, 32768, 262144, 1048576, 4194304, 8388608)

function Get-IterationsForSize([int64]$size) {
    if ($size -le 64) { return 20000 }
    if ($size -le 1024) { return 5000 }
    if ($size -le 65536) { return 2000 }
    if ($size -le 524288) { return 500 }
    if ($size -le 2097152) { return 200 }
    return 50
}

$processCount = 2
$numRuns = 5

foreach ($messageSize in $messageSizeList) {
    $numIterations = Get-IterationsForSize $messageSize
    for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
        $seed = Get-Random
        $processInfo = & mpiexec -n $processCount "$exePath" $messageSize $numIterations
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run for size $messageSize."
            continue
        }
        $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -lt 6) {
            Write-Warning "Unexpected process output (expected 6 comma-separated fields): '$processInfo'. Skipping."
            continue
        }
        # parts: [0]=messageSize, [1]=numProcesses, [2]=numIterations, [3]=totalTimeSeconds, [4]=avgRoundTripSeconds, [5]=bandwidthBytesPerSec
        $csvLine = "MPI_3,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$runIndex,PROCS=$processCount"
        $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

        Write-Host "$(Get-Date -Format 's') appended: size=$messageSize iterations=$numIterations run=$runIndex"
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
