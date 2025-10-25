#!/usr/bin/env bash
#SBATCH --job-name=MPI_11
set -euo pipefail

: "${EXE_PATH:?EXE_PATH not set}"
: "${NUM_ITERATIONS:?NUM_ITERATIONS not set}"
: "${GRID_ROWS:?GRID_ROWS not set}"
: "${GRID_COLS:?GRID_COLS not set}"
: "${RUN_INDEX:?RUN_INDEX not set}"
: "${SEED:=123456}"
: "${RESULTS_DIR:=$HOME/results}"

exePath="$EXE_PATH"
numIterations="$NUM_ITERATIONS"
gridRows="$GRID_ROWS"
gridCols="$GRID_COLS"
runIndex="$RUN_INDEX"
seed="$SEED"
resultsDir="$RESULTS_DIR"

module add openmpi >/dev/null 2>&1 || true

mkdir -p "$resultsDir"
csvPath="$resultsDir/MPI_11.csv"

tmpDir="${TMPDIR:-/tmp}"
tmpOutputFile="$tmpDir/mpi11_output_${SLURM_JOB_ID:-$$}.txt"

srun -n "${SLURM_NTASKS:-1}" "$exePath" "$numIterations" "$gridRows" "$gridCols" > "$tmpOutputFile" 2>&1 || jobExit=$?
jobExit=${jobExit:-0}

if [[ "$jobExit" -ne 0 ]]; then
    echo "MPI program failed with exit code $jobExit" >&2
    echo "=== program output (first 200 lines) ===" >&2
    sed -n '1,200p' "$tmpOutputFile" >&2
    exit "$jobExit"
fi

matchedLines="$(grep -a '^MPI_11,' "$tmpOutputFile" | tr -d '\r' || true)"

if [[ -z "$matchedLines" ]]; then
    echo "No lines starting with 'MPI_11,' found; nothing to append to CSV." >&2
    exit 2
fi

mpiEnv="SLURM_NTASKS=${SLURM_NTASKS:-1};JOBID=${SLURM_JOB_ID:-na}"

exec 9>>"$csvPath"
if command -v flock >/dev/null 2>&1; then
    flock 9
    while IFS= read -r matchedLine; do
        csvLine="$matchedLine,$runIndex,\"$mpiEnv\""
        printf '%s\n' "$csvLine" >&9
    done <<< "$matchedLines"
    flock -u 9
else
    while IFS= read -r matchedLine; do
        csvLine="$matchedLine,$runIndex,\"$mpiEnv\""
        printf '%s\n' "$csvLine" >&9
    done <<< "$matchedLines"
fi
exec 9>&-

echo "Appended $(printf '%s\n' "$matchedLines" | wc -l) lines to $csvPath"
