param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_7.exe"
$csvPath = Join-Path $resultsDir "MPI_7.csv"

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

"testType,messageSizeBytes,numProcesses,mode,numIterations,computeUnits,avgWallSeconds,avgCommSeconds,avgComputeSeconds,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$messageSizeList = @(1024, 16384, 65536, 262144, 1048576)
$processList = @(1, 2, 4, 6, 8)
$computeUnitsList = @(0, 10, 50, 200)
$numIterations = 50
$modes = @("blocking","nonblocking","comm_only","compute_only")
$numRuns = 3

foreach ($messageSize in $messageSizeList) {
    foreach ($numProcs in $processList) {
        foreach ($computeUnits in $computeUnitsList) {
            foreach ($mode in $modes) {
                for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                    $seed = Get-Random
                    $processInfo = & mpiexec -n $numProcs "$exePath" $messageSize $numIterations $computeUnits $mode $seed
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                        continue
                    }
                    $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                    if ($parts.Count -lt 9) {
                        Write-Warning "Unexpected output: '$processInfo'. Skipping."
                        continue
                    }
                    # parts correspond to CSV line from program; append runIndex and env
                    $csvLine = "MPI_7,$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$($parts[7]),$($parts[8]),$runIndex,PROCS=$numProcs"
                    $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
                    Write-Host "$(Get-Date -Format 's') appended: size=$messageSize procs=$numProcs mode=$mode units=$computeUnits run=$runIndex"
                }
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
