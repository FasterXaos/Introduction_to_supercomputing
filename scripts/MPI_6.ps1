param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_6.exe"
$csvPath = Join-Path $resultsDir "MPI_6.csv"

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

"testType,matrixSize,numProcesses,sendMode,timeSeconds,checksum,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$matrixSizeList = @(240, 480, 720, 960, 1200)
$processList = @(1, 4, 9, 16, 25)
$sendModeList = @("collective","manual_std","manual_ssend","manual_bsend","manual_rsend")
$numRuns = 5

foreach ($matrixSize in $matrixSizeList) {
    foreach ($numProcs in $processList) {
        foreach ($sendMode in $sendModeList) {
            for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                $seed = Get-Random
                $processInfo = & mpiexec -n $numProcs "$exePath" $matrixSize $sendMode $seed
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                    continue
                }
                $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -lt 5) {
                    Write-Warning "Unexpected process output (expected 5 comma-separated fields): '$processInfo'. Skipping."
                    continue
                }
                # parts: [0]=matrixSize, [1]=numProcesses, [2]=sendMode, [3]=timeSeconds, [4]=checksum
                $csvLine = "MPI_6,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$runIndex,PROCS=$numProcs"
                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

                Write-Host "$(Get-Date -Format 's') appended: N=$matrixSize procs=$numProcs mode=$($parts[2]) run=$runIndex"
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
