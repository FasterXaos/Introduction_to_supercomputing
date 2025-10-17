param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_5.exe"
$csvPath = Join-Path $resultsDir "MPI_5.csv"

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

"testType,messageSizeBytes,numMessages,computeMicroseconds,numIterations,numProcesses,totalTimeSeconds,avgTimePerIteration,bandwidthBytesPerSec,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$messageSizeList = @(1, 64, 1024, 65536, 262144)
$numMessagesList = @(1, 4, 16, 32)
$computeMicroList = @(10, 50, 100)

function Get-IterationsForSize([int64]$size) {
    if ($size -le 64) { return 2000 }
    if ($size -le 1024) { return 1000 }
    if ($size -le 65536) { return 500 }
    if ($size -le 262144) { return 200 }
    return 100
}

$processList = @(2, 4, 6, 8)
$numRuns = 5

foreach ($messageSize in $messageSizeList) {
    $numIterations = Get-IterationsForSize $messageSize
    foreach ($numMessages in $numMessagesList) {
        foreach ($computeMicro in $computeMicroList) {
            foreach ($procs in $processList) {
                for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                    $processInfo = & mpiexec -n $procs "$exePath" $messageSize $numMessages $computeMicro $numIterations sleep
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                        continue
                    }
                    $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                    if ($parts.Count -lt 8) {
                        Write-Warning "Unexpected process output (expected 8 comma-separated fields): '$processInfo'. Skipping."
                        continue
                    }
                    # parts: [0]=messageSizeBytes, [1]=numMessages, [2]=computeMicroseconds, [3]=numIterations, [4]=numProcesses, [5]=totalTimeSeconds, [6]=avgTimePerIteration, [7]=bandwidthBytesPerSec
                    $csvLine = "MPI_5,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$($parts[7]),$runIndex,PROCS=$procs"
                    $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
                    Write-Host "$(Get-Date -Format 's') appended: size=$messageSize msgs=$numMessages computeUs=$computeMicro procs=$procs run=$runIndex"
                }
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
