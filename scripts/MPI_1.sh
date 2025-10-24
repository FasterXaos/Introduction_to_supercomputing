#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_1"
jobScript="$scriptDir/MPI_1_job.sh"
csvPath="$resultsDir/MPI_1.csv"

vectorSizeList=(1000000 5000000 10000000)
processList=(1 2 4 6 8 16 32)
modeList=("min" "max")
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_1.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_1.cpp" ]]; then
    echo "Source file not found: $srcDir/MPI_1.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_1.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Compilation failed or executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built executable: $binDir/$exeName"

printf '%s\n' "testType,vectorSize,numProcesses,mode,timeSeconds,resultValue,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for mode in "${modeList[@]}"; do
    for vectorSize in "${vectorSizeList[@]}"; do
        for procs in "${processList[@]}"; do
            for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                seed=$RANDOM

                sbatch --ntasks="$procs" \
                       --output="$logDir/MPI_1-%j.out" \
                       --error="$logDir/MPI_1-%j.err" \
                       --export=ALL,EXE_PATH="$binDir/$exeName",VECTOR_SIZE="$vectorSize",MODE="$mode",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                       --parsable \
                       "$jobScript" >/dev/null

                echo "$(date -Is) queued: mode=$mode size=$vectorSize procs=$procs run=$runIndex"
                # sleep 0.05
            done
        done
    done
done

echo "All jobs submitted. Results will be appended to $csvPath"
