#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_5"
jobScript="$scriptDir/MPI_5_job.sh"
csvPath="$resultsDir/MPI_5.csv"

messageSizeList=(1 64 1024 65536 262144)
numMessagesList=(1 4 16 32)
computeMicroList=(10 50 100)

processList=(2 4 6 8)
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_5.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_5.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_5.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_5.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

function getIterationsForSize() {
    local size=$1
    if (( size <= 64 )); then
        echo 2000
    elif (( size <= 1024 )); then
        echo 1000
    elif (( size <= 65536 )); then
        echo 500
    elif (( size <= 262144 )); then
        echo 200
    else
        echo 100
    fi
}

printf '%s\n' "testType,messageSizeBytes,numMessages,computeMicroseconds,numIterations,numProcesses,totalTimeSeconds,avgTimePerIteration,bandwidthBytesPerSec,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for messageSize in "${messageSizeList[@]}"; do
    numIterations="$(getIterationsForSize "$messageSize")"
    for numMessages in "${numMessagesList[@]}"; do
        for computeMicro in "${computeMicroList[@]}"; do
            for procs in "${processList[@]}"; do
                for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                    seed=$RANDOM
                    computeMode="sleep"
                    sbatch --ntasks="$procs" \
                           --output="$logDir/MPI_5-%j.out" \
                           --error="$logDir/MPI_5-%j.err" \
                           --export=ALL,EXE_PATH="$binDir/$exeName",MESSAGE_SIZE="$messageSize",NUM_MESSAGES="$numMessages",COMPUTE_MICROS="$computeMicro",NUM_ITERATIONS="$numIterations",COMPUTE_MODE="$computeMode",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                           --parsable \
                           "$jobScript" >/dev/null

                    echo "$(date -Is) queued: size=$messageSize msgs=$numMessages computeUs=$computeMicro procs=$procs run=$runIndex"
                    # sleep 0.05
                done
            done
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
