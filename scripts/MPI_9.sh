#!/usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_9"
jobScript="$scriptDir/MPI_9_job.sh"
csvPath="$resultsDir/MPI_9.csv"

opList=(bcast reduce scatter gather allgather alltoall)
messageSizeList=(1 16 1024 16384 65536 262144 1048576)
processList=(1 2 4 8 16)
numRuns=3

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_9.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_9.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_9.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_9.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

printf '%s\n' "testType,opName,messageSizeBytes,numProcesses,customTime,mpiTime,checksum,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for op in "${opList[@]}"; do
    for messageSize in "${messageSizeList[@]}"; do
        for numProcs in "${processList[@]}"; do
            for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                seed=$RANDOM
                sbatch --ntasks="$numProcs" \
                       --output="$logDir/MPI_9-%j.out" \
                       --error="$logDir/MPI_9-%j.err" \
                       --export=ALL,EXE_PATH="$binDir/$exeName",OP_NAME="$op",MESSAGE_SIZE="$messageSize",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                       --parsable \
                       "$jobScript" >/dev/null

                echo "$(date -Is) queued: op=$op msg=$messageSize procs=$numProcs run=$runIndex"
                # sleep 0.05
            done
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
