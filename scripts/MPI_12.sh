#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_12"
jobScript="$scriptDir/MPI_12_job.sh"
csvPath="$resultsDir/MPI_12.csv"

processList=(2 4 6 8 9 16 32 64 128)
numIterations=100
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_12.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_12.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_12.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_12.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

printf '%s\n' "testType,topology,gridRows,gridCols,numProcesses,commCreated,avgTimePerAllreduce,finalGlobal,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for numProcs in "${processList[@]}"; do
    approx=$(awk "BEGIN{printf(\"%d\", int(sqrt($numProcs)))}")
    gridRows=1
    for (( r=approx; r>=1; r-- )); do
        if (( numProcs % r == 0 )); then
            gridRows=$r
            break
        fi
    done
    gridCols=$(( numProcs / gridRows ))

    for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
        seed=$RANDOM
        sbatch --ntasks="$numProcs" \
               --output="$logDir/MPI_12-%j.out" \
               --error="$logDir/MPI_12-%j.err" \
               --export=ALL,EXE_PATH="$binDir/$exeName",NUM_ITERATIONS="$numIterations",GRID_ROWS="$gridRows",GRID_COLS="$gridCols",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
               --parsable \
               "$jobScript" >/dev/null

        echo "$(date -Is) queued: procs=$numProcs grid=${gridRows}x${gridCols} run=$runIndex"
        # sleep 0.05
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
