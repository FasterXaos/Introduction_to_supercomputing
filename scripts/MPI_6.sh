#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_6"
jobScript="$scriptDir/MPI_6_job.sh"
csvPath="$resultsDir/MPI_6.csv"

matrixSizeList=(240 480 720 960 1200)
processList=(1 4 9 16 25)
sendModeList=("collective" "manual_std" "manual_ssend" "manual_bsend" "manual_rsend")
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_6.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_6.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_6.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_6.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

printf '%s\n' "testType,matrixSize,numProcesses,sendMode,timeSeconds,checksum,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for matrixSize in "${matrixSizeList[@]}"; do
    for numProcs in "${processList[@]}"; do
        for sendMode in "${sendModeList[@]}"; do
            for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                seed=$RANDOM
                sbatch --ntasks="$numProcs" \
                       --output="$logDir/MPI_6-%j.out" \
                       --error="$logDir/MPI_6-%j.err" \
                       --export=ALL,EXE_PATH="$binDir/$exeName",MATRIX_SIZE="$matrixSize",SEND_MODE="$sendMode",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                       --parsable \
                       "$jobScript" >/dev/null

                echo "$(date -Is) queued: N=$matrixSize procs=$numProcs mode=$sendMode run=$runIndex"
                # sleep 0.05
            done
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
