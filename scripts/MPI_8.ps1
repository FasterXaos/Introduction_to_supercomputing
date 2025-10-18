param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "MPI_8.exe"
$csvPath = Join-Path $resultsDir "MPI_8.csv"

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

"testType,messageSize,numProcesses,mode,numIterations,totalTime,avgRoundTrip,bandwidth,runIndex,mpiEnv" | Out-File -FilePath $csvPath -Encoding utf8

$messageSizeList = @(1, 16, 1024, 16384, 65536, 262144, 1048576)
$modes = @("separate","sendrecv","isend_irecv")
$numRuns = 5

foreach ($messageSize in $messageSizeList) {
    if ($messageSize -le 64) { $numIterations = 20000 }
    elseif ($messageSize -le 1024) { $numIterations = 5000 }
    elseif ($messageSize -le 65536) { $numIterations = 2000 }
    elseif ($messageSize -le 524288) { $numIterations = 500 }
    elseif ($messageSize -le 2097152) { $numIterations = 200 }
    else { $numIterations = 50 }

    foreach ($mode in $modes) {
        for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
            $processInfo = & mpiexec -n 2 "$exePath" $messageSize $mode $numIterations
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                continue
            }
            $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
            if ($parts.Count -lt 8) {
                Write-Warning "Unexpected output: '$processInfo'. Skipping."
                continue
            }
            # parts: [0]=MPI_8, [1]=messageSize, [2]=numProcesses, [3]=mode, [4]=numIterations, [5]=totalTime, [6]=avgRoundTrip, [7]=bandwidth
            $csvLine = "MPI_8,$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$($parts[7]),$runIndex,PROCS=2"
            $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8
            Write-Host "$(Get-Date -Format 's') appended: size=$messageSize mode=$mode run=$runIndex"
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
