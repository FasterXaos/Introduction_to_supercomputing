param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_9.exe"
$csvPath = Join-Path $resultsDir "MPI_9.csv"

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

"testType,opName,messageSizeBytes,numProcesses,customTime,mpiTime,checksum,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$opList = @("bcast","reduce","scatter","gather","allgather","alltoall")
$messageSizeList = @(1, 16, 1024, 16384, 65536, 262144, 1048576)
$processList = @(1, 2, 4, 8)
$numRuns = 3

foreach ($op in $opList) {
    foreach ($msgSize in $messageSizeList) {
        foreach ($procs in $processList) {
            for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                $processInfo = & mpiexec -n $procs "$exePath" $op $msgSize
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping."
                    continue
                }
                $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -lt 9) {
                    Write-Warning "Unexpected output: '$processInfo'. Skipping."
                    continue
                }
                # parts: [0]=MPI_9, [1]=opName, [2]=messageSize, [3]=numProcesses, [4]=customTime, [5]=mpiTime, [6]=checksum, [7]=runIndex, [8]=mpiEnv
                $csvLine = "MPI_9,$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$runIndex,PROCS=$procs"
                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
                Write-Host "$(Get-Date -Format 's') appended: op=$op msg=$msgSize procs=$procs run=$runIndex"
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
