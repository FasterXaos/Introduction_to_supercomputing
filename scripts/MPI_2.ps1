param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_2.exe"
$csvPath = Join-Path $resultsDir "MPI_2.csv"

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

"testType,problemSize,numProcesses,timeSeconds,dotProduct,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$problemSizeList = @(1000000, 5000000, 10000000)
$processList = @(1, 2, 4, 6, 8, 16, 32)
$numRuns = 5

foreach ($problemSize in $problemSizeList) {
    foreach ($procs in $processList) {
        for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
            $seed = Get-Random
            $processInfo = & mpiexec -n $procs "$exePath" $problemSize $seed
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                continue
            }

            $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
            if ($parts.Count -lt 4) {
                Write-Warning "Unexpected process output (expected 4 comma-separated fields): '$processInfo'. Skipping."
                continue
            }

            # parts: [0]=problemSize, [1]=numProcesses, [2]=timeSeconds, [3]=dotProduct
            $csvLine = "MPI_2,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$runIndex,PROCS=$procs"
            $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

            Write-Host "$(Get-Date -Format 's') appended: size=$problemSize procs=$procs run=$runIndex"
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
