param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "OpenMP_3.exe"
$csvPath = Join-Path $resultsDir "OpenMP_3.csv"

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

"testType,numIntervals,numThreads,mode,timeSeconds,integralValue,runIndex,ompEnv" | Out-File -FilePath $csvPath -Encoding utf8

$problemSizeList = @(1000000, 5000000, 10000000, 50000000)
$threadList = @(1, 2, 4, 6, 8, 16, 32, 64)
$modeList = @("reduction", "no_reduction")
$numRuns = 5

$lowerBound = 0.0
$upperBound = 3.141592653589793

foreach ($mode in $modeList) {
    foreach ($numIntervals in $problemSizeList) {
        foreach ($threads in $threadList) {
            $env:OMP_NUM_THREADS = "$threads"
            for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                $processInfo = & "$exePath" $numIntervals $mode $lowerBound $upperBound
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                    continue
                }

                $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                if ($parts.Count -lt 5) {
                    Write-Warning "Unexpected process output (expected 5 comma-separated fields): '$processInfo'. Skipping."
                    continue
                }

                # parts: [0]=numIntervals, [1]=numThreads, [2]=mode, [3]=timeSeconds, [4]=integralValue
                $csvLine = "OpenMP_3,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$runIndex,OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

                Write-Host "$(Get-Date -Format 's') appended: mode=$mode intervals=$numIntervals threads=$threads run=$runIndex"
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
