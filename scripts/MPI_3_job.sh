#!/usr/bin/env bash
#SBATCH --job-name=MPI_3
set -euo pipefail

: "${EXE_PATH:?EXE_PATH not set}"
: "${MESSAGE_SIZE:?MESSAGE_SIZE not set}"
: "${NUM_ITERATIONS:?NUM_ITERATIONS not set}"
: "${RUN_INDEX:?RUN_INDEX not set}"
: "${SEED:=123456}"
: "${RESULTS_DIR:=$HOME/results}"

module add openmpi >/dev/null 2>&1 || true

mkdir -p "$RESULTS_DIR"
csvPath="$RESULTS_DIR/MPI_3.csv"

tmpDir="${TMPDIR:-/tmp}"
tmpOutputFile="$tmpDir/mpi3_output_${SLURM_JOB_ID:-$$}.txt"

srun -n "${SLURM_NTASKS:-2}" "$EXE_PATH" "$MESSAGE_SIZE" "$NUM_ITERATIONS" > "$tmpOutputFile" 2>&1 || jobExit=$?
jobExit=${jobExit:-0}

if [[ "$jobExit" -ne 0 ]]; then
    echo "MPI program failed with exit code $jobExit" >&2
    echo "=== program output (first 200 lines) ===" >&2
    sed -n '1,200p' "$tmpOutputFile" >&2
    exit "$jobExit"
fi

outputLine="$(sed -n '/\S/p' "$tmpOutputFile" | sed -n '1p' | tr -d '\r' || true)"

if [[ -z "$outputLine" ]]; then
    echo "Empty program output; nothing to append to CSV." >&2
    exit 2
fi

mpiEnv="SLURM_NTASKS=${SLURM_NTASKS:-2};JOBID=${SLURM_JOB_ID:-na}"
csvLine="MPI_3,$outputLine,$RUN_INDEX,\"$mpiEnv\""

exec 9>>"$csvPath"
if command -v flock >/dev/null 2>&1; then
    flock 9
    printf '%s\n' "$csvLine" >&9
    flock -u 9
else
    printf '%s\n' "$csvLine" >&9
fi
exec 9>&-

echo "Appended to $csvPath: $csvLine"
