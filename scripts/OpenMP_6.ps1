param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "OpenMP_6.exe"
$csvPath = Join-Path $resultsDir "OpenMP_6.csv"

New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

Write-Host "Configuring and building via CMake..."
& cmake -S $ProjectRoot -B $buildDir -DCMAKE_BUILD_TYPE=Release
& cmake --build $buildDir --config Release

$exePath = Join-Path $binDir $exeName

if (-not (Test-Path $exePath)) {
    Write-Error "Executable not found at $exePath. Проверьте, что CMakeLists.txt находится в $ProjectRoot и сборка прошла успешно."
    exit 1
}

$metadataPath = Join-Path $resultsDir "metadata.txt"
if (-not (Test-Path $metadataPath)) {
    $compilerInfo = & cmake --version | Select-Object -First 1
    $osInfo = (Get-CimInstance Win32_OperatingSystem).Caption
    "$compilerInfo" | Out-File -FilePath $metadataPath -Encoding utf8
    "os: $osInfo" | Out-File -FilePath $metadataPath -Append -Encoding utf8
    "buildDate: $(Get-Date -Format o)" | Out-File -FilePath $metadataPath -Append -Encoding utf8
}

"testType,problemSize,numThreads,schedule,chunk,timeSeconds,resultSum,heavyProbability,lightWork,heavyWork,runIndex,ompEnv" | Out-File -FilePath $csvPath -Encoding utf8

$problemSizeList = @(10000, 50000, 100000)
$threadList = @(1, 2, 4, 6, 8, 16, 32, 64)
$scheduleList = @("static","dynamic","guided")
$chunkList = @(1, 10, 100)


$heavyProbabilityList = @(0.05, 0.15)
$lightWork = 10
$heavyWork = 1000

$numRuns = 5

foreach ($problemSize in $problemSizeList) {
    foreach ($heavyProb in $heavyProbabilityList) {
        foreach ($schedule in $scheduleList) {
            foreach ($chunk in $chunkList) {
                foreach ($threads in $threadList) {
                    $env:OMP_NUM_THREADS = "$threads"
                    for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                        $seed = Get-Random
                        $processInfo = & "$exePath" $problemSize $schedule $chunk $heavyProb $lightWork $heavyWork $seed
                        if ($LASTEXITCODE -ne 0) {
                            Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                            continue
                        }

                        $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                        if ($parts.Count -lt 6) {
                            Write-Warning "Unexpected process output (expected 6 comma-separated fields): '$processInfo'. Skipping."
                            continue
                        }

                        # parts: [0]=problemSize, [1]=numThreads, [2]=schedule, [3]=chunk, [4]=timeSeconds, [5]=resultSum
                        $csvLine = "OpenMP_6,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$heavyProb,$lightWork,$heavyWork,$runIndex,OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
                        $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

                        Write-Host "$(Get-Date -Format 's') appended: N=$problemSize heavyProb=$heavyProb schedule=$schedule chunk=$chunk threads=$threads run=$runIndex"
                    }
                }
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
