param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_11.exe"
$csvPath = Join-Path $resultsDir "MPI_11.csv"

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

"testType,gridRows,gridCols,numProcesses,commType,medianTimeSeconds,globalSum,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

function Get-GridDims([int]$procCount) {
    $approx = [math]::Floor([math]::Sqrt($procCount))
    for ($r = $approx; $r -ge 1; $r--) {
        if ($procCount % $r -eq 0) {
            return ,$r, ($procCount / $r)
        }
    }
    return ,1,$procCount
}

$processList = @(1, 2, 4, 6, 8, 9, 16, 32, 64)
$numIterationsList = @(200, 500)
$numRuns = 3

foreach ($numProcesses in $processList) {
    $gridDims = Get-GridDims $numProcesses
    $gridRows = $gridDims[0]
    $gridCols = $gridDims[1]

    foreach ($numIterations in $numIterationsList) {
        for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
            $processInfo = & mpiexec -n $numProcesses "$exePath" $numIterations $gridRows $gridCols
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                continue
            }

            $lines = $processInfo -split "`n" | Where-Object { $_ -ne "" }
            foreach ($line in $lines) {
                $parts = ($line -split ',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -lt 7) {
                    Write-Warning "Unexpected output: '$line'"
                    continue
                }
                # parts: [0]=MPI_11, [1]=gridRows, [2]=gridCols, [3]=numProcesses, [4]=commType, [5]=medianTimeSeconds, [6]=globalSum
                $csvLine = "MPI_11,$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$runIndex,PROCS=$numProcesses"
                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
                Write-Host "$(Get-Date -Format 's') appended: procs=$numProcesses grid=${gridRows}x${gridCols} comm=$($parts[4]) iters=$numIterations run=$runIndex"
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
