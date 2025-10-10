param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "OpenMP_8.exe"
$csvPath = Join-Path $resultsDir "OpenMP_8.csv"

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
    $compilerInfo = & cmake --version | Select-Object -First 1
    $osInfo = (Get-CimInstance Win32_OperatingSystem).Caption
    "$compilerInfo" | Out-File -FilePath $metadataPath -Encoding utf8
    "os: $osInfo" | Out-File -FilePath $metadataPath -Append -Encoding utf8
    "buildDate: $(Get-Date -Format o)" | Out-File -FilePath $metadataPath -Append -Encoding utf8
}

"testType,numVectors,vectorSize,numThreads,mode,timeSeconds,totalSum,runIndex,ompEnv" | Out-File -FilePath $csvPath -Encoding utf8

$vectorCountList = @(10, 50)
$vectorSizeList = @(100000, 300000)
$threadList = @(1, 2, 4, 6, 8, 16, 32)
$modeList = @("sequential", "sections")
$numRuns = 5

foreach ($vectorCount in $vectorCountList) {
    foreach ($vectorSize in $vectorSizeList) {
        foreach ($mode in $modeList) {
            foreach ($threads in $threadList) {
                $env:OMP_NUM_THREADS = "$threads"
                for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                    $seed = Get-Random
                    $processInfo = & "$exePath" $vectorCount $vectorSize $mode $seed
                    if ($LASTEXITCODE -ne 0) {
                        Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                        continue
                    }

                    $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                    if ($parts.Count -lt 6) {
                        Write-Warning "Unexpected process output (expected 6 comma-separated fields): '$processInfo'. Skipping."
                        continue
                    }

                    # parts: [0]=numVectors, [1]=vectorSize, [2]=numThreads, [3]=mode, [4]=timeSeconds, [5]=totalSum
                    $csvLine = "OpenMP_8,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$runIndex,OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
                    $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

                    Write-Host "$(Get-Date -Format 's') appended: vectors=$vectorCount size=$vectorSize mode=$mode threads=$threads run=$runIndex"
                }
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
