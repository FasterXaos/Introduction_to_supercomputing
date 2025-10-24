#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_2"
jobScript="$scriptDir/MPI_2_job.sh"
csvPath="$resultsDir/MPI_2.csv"

problemSizeList=(1000000 5000000 10000000)
processList=(1 2 4 6 8 16 32)
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_2.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_2.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_2.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_2.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

echo "Creating fresh CSV: $csvPath"
printf '%s\n' "testType,problemSize,numProcesses,timeSeconds,dotProduct,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for problemSize in "${problemSizeList[@]}"; do
    for procs in "${processList[@]}"; do
        for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
            seed=$RANDOM
            sbatch --ntasks="$procs" \
                   --output="$logDir/MPI_2-%j.out" \
                   --error="$logDir/MPI_2-%j.err" \
                   --export=ALL,EXE_PATH="$binDir/$exeName",PROBLEM_SIZE="$problemSize",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                   --parsable \
                   "$jobScript" >/dev/null

            echo "$(date -Is) queued: size=$problemSize procs=$procs run=$runIndex"
            # sleep 0.05
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
