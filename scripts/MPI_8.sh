#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_8"
jobScript="$scriptDir/MPI_8_job.sh"
csvPath="$resultsDir/MPI_8.csv"

messageSizeList=(1 16 1024 16384 65536 262144 1048576 4194304)
modes=("separate" "sendrecv" "isend_irecv")
numRuns=5
processCount=2

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_8.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_8.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_8.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_8.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

function getIterationsForSize() {
    local size=$1
    if (( size <= 64 )); then
        echo 20000
    elif (( size <= 1024 )); then
        echo 5000
    elif (( size <= 65536 )); then
        echo 2000
    elif (( size <= 524288 )); then
        echo 500
    elif (( size <= 2097152 )); then
        echo 200
    else
        echo 50
    fi
}

printf '%s\n' "testType,messageSize,numProcesses,mode,numIterations,totalTime,avgRoundTrip,bandwidth,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for messageSize in "${messageSizeList[@]}"; do
    numIterations="$(getIterationsForSize "$messageSize")"
    for mode in "${modes[@]}"; do
        for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
            seed=$RANDOM
            sbatch --ntasks="$processCount" \
                   --output="$logDir/MPI_8-%j.out" \
                   --error="$logDir/MPI_8-%j.err" \
                   --export=ALL,EXE_PATH="$binDir/$exeName",MESSAGE_SIZE="$messageSize",MODE="$mode",NUM_ITERATIONS="$numIterations",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                   --parsable \
                   "$jobScript" >/dev/null

            echo "$(date -Is) queued: size=$messageSize mode=$mode run=$runIndex"
            # sleep 0.05
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
