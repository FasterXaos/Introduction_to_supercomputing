param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_10.exe"
$csvPath = Join-Path $resultsDir "MPI_10.csv"

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

"testType,method,matrixRows,matrixCols,blockRows,blockCols,numProcesses,timeSeconds,checksum,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$matrixSizes = @(512, 1024, 2048, 4096)
$blockPairs = @(
    @{ rows = 32; cols = 32 },
    @{ rows = 64; cols = 64 }
)
$methods = @("derived","pack","manual")
$processList = @(1, 2, 4, 6, 8)
$numRuns = 5

foreach ($matrixSize in $matrixSizes) {
    foreach ($pair in $blockPairs) {
        $blockRows = $pair.rows
        $blockCols = $pair.cols
        foreach ($procs in $processList) {
            foreach ($method in $methods) {
                for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                    $seed = Get-Random
                    $processInfo = & mpiexec -n $procs "$exePath" $matrixSize $matrixSize $blockRows $blockCols $method $seed
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                        continue
                    }
                    $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                    if ($parts.Count -lt 8) {
                        Write-Warning "Unexpected process output (expected at least 8 comma-separated fields): '$processInfo'. Skipping."
                        continue
                    }
                    # parts: [0]=method, [1]=matrixRows, [2]=matrixCols, [3]=blockRows, [4]=blockCols, [5]=numProcesses, [6]=timeSeconds, [7]=checksum
                    $csvLine = "MPI_10,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$($parts[7]),$runIndex,PROCS=$procs"
                    $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
                    Write-Host "$(Get-Date -Format 's') appended: method=$method N=$matrixSize block=${blockRows}x${blockCols} procs=$procs run=$runIndex"
                }
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
