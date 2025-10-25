#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_7"
jobScript="$scriptDir/MPI_7_job.sh"
csvPath="$resultsDir/MPI_7.csv"

messageSizeList=(1024 16384 65536 262144 1048576)
processList=(1 2 4 6 8)
computeUnitsList=(0 10 50 200)
numIterations=50
modes=("blocking" "nonblocking" "comm_only" "compute_only")
numRuns=3

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_7.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_7.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_7.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_7.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

printf '%s\n' "testType,messageSizeBytes,numProcesses,mode,numIterations,computeUnits,avgWallSeconds,avgCommSeconds,avgComputeSeconds,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for messageSize in "${messageSizeList[@]}"; do
    for procs in "${processList[@]}"; do
        for computeUnits in "${computeUnitsList[@]}"; do
            for mode in "${modes[@]}"; do
                for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                    seed=$RANDOM
                    sbatch --ntasks="$procs" \
                           --output="$logDir/MPI_7-%j.out" \
                           --error="$logDir/MPI_7-%j.err" \
                           --export=ALL,EXE_PATH="$binDir/$exeName",MESSAGE_SIZE="$messageSize",NUM_ITERATIONS="$numIterations",COMPUTE_UNITS="$computeUnits",MODE="$mode",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                           --parsable \
                           "$jobScript" >/dev/null

                    echo "$(date -Is) queued: size=$messageSize procs=$procs mode=$mode units=$computeUnits run=$runIndex"
                    # sleep 0.05
                done
            done
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
