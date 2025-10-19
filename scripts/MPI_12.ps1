param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_12.exe"
$csvPath = Join-Path $resultsDir "MPI_12.csv"

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

"testType,topology,gridRows,gridCols,numProcesses,commCreated,avgTimePerAllreduce,finalGlobal,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$processList = @(2, 4, 6, 8, 9, 16, 32)
$numIterations = 200
$numRuns = 3

foreach ($numProcs in $processList) {
    for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
        $processInfo = & mpiexec -n $numProcs "$exePath" $numIterations
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
            continue
        }

        $lines = $processInfo -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
        foreach ($line in $lines) {
            if ($line.StartsWith("MPI_12,")) {
                $csvLine = "$line,$runIndex,PROCS=$numProcs"
                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
                Write-Host "$(Get-Date -Format 's') appended: procs=$numProcs line=$line"
            } else {
                Write-Host $line
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
