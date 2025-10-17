param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_4.exe"
$csvPath = Join-Path $resultsDir "MPI_4.csv"

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

"testType,matrixSize,numProcesses,mode,timeSeconds,checksum,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$matrixSizeList = @(240, 480, 720, 960, 1200)
$processList = @(1, 4, 9, 16, 25)
$modeList = @("blockRow","cannon")
$numRuns = 5

foreach ($matrixSize in $matrixSizeList) {
    foreach ($numProcs in $processList) {
        foreach ($mode in $modeList) {
            $sqrtP = [math]::Sqrt($numProcs)
            $isPerfectSquare = ([math]::Round($sqrtP) * [math]::Round($sqrtP) - $numProcs) -eq 0
            $canUseCannon = $false
            if ($mode -eq "cannon") {
                if ($isPerfectSquare) {
                    $q = [int][math]::Round($sqrtP)
                    if (($matrixSize % $q) -eq 0) { $canUseCannon = $true }
                }
                if (-not $canUseCannon) {
                    Write-Warning "Skipping cannon for matrixSize=$matrixSize procs=$numProcs (requirements not met). Will run blockRow instead."
                }
            }

            $effectiveMode = $mode
            if ($mode -eq "cannon" -and -not $canUseCannon) {
                $effectiveMode = "blockRow"
            }

            for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                $seed = Get-Random
                $processInfo = & mpiexec -n $numProcs "$exePath" $matrixSize $effectiveMode $seed
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                    continue
                }
                $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -lt 5) {
                    Write-Warning "Unexpected process output (expected 5 comma-separated fields): '$processInfo'. Skipping."
                    continue
                }
                # parts: [0]=matrixSize, [1]=numProcesses, [2]=mode, [3]=timeSeconds, [4]=checksum
                $csvLine = "MPI_4,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$runIndex,PROCS=$numProcs"
                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

                Write-Host "$(Get-Date -Format 's') appended: N=$matrixSize procs=$numProcs mode=$($parts[2]) run=$runIndex"
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
