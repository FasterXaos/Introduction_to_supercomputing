#!/usr/bin/env bash
set -euo pipefail

# Пути
scriptDir="$(cd "$(dirname "$0")" && pwd)"
projectRoot="$(cd "$scriptDir/.." && pwd)"

srcDir="$projectRoot/src"
buildDir="$projectRoot/build"
binDir="$buildDir/bin"
resultsDir="$projectRoot/results"
logDir="$resultsDir/logs"
exeName="MPI_4"
jobScript="$scriptDir/MPI_4_job.sh"
csvPath="$resultsDir/MPI_4.csv"

matrixSizeList=(240 480 720 960 1200)
processList=(1 4 9 16 25)
modeList=("blockRow" "cannon")
numRuns=5

mkdir -p "$binDir"
mkdir -p "$resultsDir"
mkdir -p "$logDir"

module add openmpi >/dev/null 2>&1 || true

echo "Compiling $srcDir/MPI_4.cpp -> $binDir/$exeName"
if [[ ! -f "$srcDir/MPI_4.cpp" ]]; then
    echo "Source not found: $srcDir/MPI_4.cpp" >&2
    exit 1
fi

mpicxx -O3 -std=c++17 -march=native -o "$binDir/$exeName" "$srcDir/MPI_4.cpp"

if [[ ! -x "$binDir/$exeName" ]]; then
    echo "Build failed: executable not found at $binDir/$exeName" >&2
    exit 2
fi
echo "Built: $binDir/$exeName"

printf '%s\n' "testType,matrixSize,numProcesses,mode,timeSeconds,checksum,runIndex,mpiEnv" > "$csvPath"

echo "Submitting jobs to Slurm (logs -> $logDir)..."
for matrixSize in "${matrixSizeList[@]}"; do
    for numProcs in "${processList[@]}"; do
        for mode in "${modeList[@]}"; do
            effectiveMode="$mode"
            if [[ "$mode" == "cannon" ]]; then
                qVal="$(awk -v p="$numProcs" 'BEGIN{q=int(sqrt(p)+0.5); print q}')"
                if ! ( (( qVal * qVal == numProcs )) && (( matrixSize % qVal == 0 )) ); then
                    effectiveMode="blockRow"
                    echo "Warning: skipping cannon for matrixSize=$matrixSize procs=$numProcs (requirements not met); using blockRow"
                fi
            fi

            for (( runIndex=1; runIndex<=numRuns; runIndex++ )); do
                seed=$RANDOM
                sbatch --ntasks="$numProcs" \
                       --output="$logDir/MPI_4-%j.out" \
                       --error="$logDir/MPI_4-%j.err" \
                       --export=ALL,EXE_PATH="$binDir/$exeName",MATRIX_SIZE="$matrixSize",MODE="$effectiveMode",RUN_INDEX="$runIndex",SEED="$seed",RESULTS_DIR="$resultsDir" \
                       --parsable \
                       "$jobScript" >/dev/null

                echo "$(date -Is) queued: N=$matrixSize procs=$numProcs mode=$effectiveMode run=$runIndex"
                # sleep 0.05
            done
        done
    done
done

echo "All jobs submitted. Fresh CSV at: $csvPath"
