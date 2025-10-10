param(
    [string]$ProjectRoot = (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$buildDir = Join-Path $ProjectRoot "build"
$binDir = Join-Path $buildDir "bin"
$resultsDir = Join-Path $ProjectRoot "results"
$exeName = "OpenMP_5.exe"
$csvPath = Join-Path $resultsDir "OpenMP_5.csv"

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

"testType,matrixSize,numThreads,mode,matrixType,bandwidth,schedule,chunk,timeSeconds,maxOfRowMins,runIndex,ompEnv" | Out-File -FilePath $csvPath -Encoding utf8

$matrixSizeList = @(500, 1000, 2000, 4000)
$threadList = @(1, 2, 4, 6, 8, 16, 32)
$modeList = @("reduction", "no_reduction")
$scheduleList = @("static", "dynamic", "guided")
$chunkList = @(1, 16)
$matrixTypeList = @("banded", "triangular")
$bandwidthList = @(3, 16)

$numRuns = 5

foreach ($mode in $modeList) {
    foreach ($matrixType in $matrixTypeList) {
        foreach ($matrixSize in $matrixSizeList) {
            foreach ($bandwidth in $bandwidthList) {
                foreach ($schedule in $scheduleList) {
                    foreach ($chunk in $chunkList) {
                        foreach ($threads in $threadList) {
                            $env:OMP_NUM_THREADS = "$threads"
                            for ($runIndex = 1; $runIndex -le $numRuns; $runIndex++) {
                                $seed = Get-Random
                                $processInfo = & "$exePath" $matrixSize $mode $matrixType $schedule $chunk $bandwidth $seed
                                if ($LASTEXITCODE -ne 0) {
                                    Write-Warning "Process returned non-zero exit code ($LASTEXITCODE). Skipping this run."
                                    continue
                                }

                                $parts = ($processInfo -split ',') | ForEach-Object { $_.Trim() }
                                if ($parts.Count -lt 9) {
                                    Write-Warning "Unexpected process output (expected >=9 comma-separated fields): '$processInfo'. Skipping."
                                    continue
                                }

                                # parts: [0]=matrixSize, [1]=numThreads, [2]=mode, [3]=matrixType, [4]=bandwidth, [5]=schedule, [6]=chunk, [7]=timeSeconds, [8]=maxOfRowMins
                                $csvLine = "OpenMP_5,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4]),$($parts[5]),$($parts[6]),$($parts[7]),$($parts[8]),$runIndex,OMP_NUM_THREADS=$($env:OMP_NUM_THREADS)"
                                $csvLine | Out-File -FilePath $csvPath -Append -Encoding utf8

                                Write-Host "$(Get-Date -Format 's') appended: mode=$mode type=$matrixType size=$matrixSize band=$bandwidth schedule=$schedule chunk=$chunk threads=$threads run=$runIndex"
                            }
                        }
                    }
                }
            }
        }
    }
}

Write-Host "Sweep finished. Results written to $csvPath"
