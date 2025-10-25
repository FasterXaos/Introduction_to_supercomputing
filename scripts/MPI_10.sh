#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_10"
jobScript="$scriptDir/MPI_10_job.sh"
csvPath="$resultsDir/MPI_10.csv"

matrixSizes=(512 1024 2048 4096)
blockPairs=("32x32" "64x64")
processList=(2 4 6 8 16)
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_10.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_10.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_10.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_10.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

printf '%s\n' "testType,method,matrixRows,matrixCols,blockRows,blockCols,numProcesses,timeSeconds,checksum,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for matrixSize in "${matrixSizes[@]}"; do
    for pair in "${blockPairs[@]}"; do
        blockRows="${pair%%x*}"
        blockCols="${pair##*x}"
        for numProcs in "${processList[@]}"; do
            for method in "derived" "pack" "manual"; do
                for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                    seed=$RANDOM
                    sbatch --ntasks="$numProcs" \
                           --output="$logDir/MPI_10-%j.out" \
                           --error="$logDir/MPI_10-%j.err" \
                           --export=ALL,EXE_PATH="$binDir/$exeName",MATRIX_ROWS="$matrixSize",MATRIX_COLS="$matrixSize",BLOCK_ROWS="$blockRows",BLOCK_COLS="$blockCols",METHOD="$method",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                           --parsable \
                           "$jobScript" >/dev/null

                    echo "$(date -Is) queued: method=$method N=${matrixSize}x${matrixSize} block=${blockRows}x${blockCols} procs=$numProcs run=$runIndex"
                    # sleep 0.05
                done
            done
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
