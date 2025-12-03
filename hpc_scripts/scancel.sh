#!/usr/bin/env bash
# scancel.sh - cancel SLURM jobs in a numeric range
# Usage: scancel.sh START_ID END_ID

set -eu

if [ $# -ne 2 ]; then
    echo "Usage: $0 START_ID END_ID" >&2
    exit 2
fi

start=$1
end=$2

re='^[0-9]+$'
if ! [[ $start =~ $re && $end =~ $re ]]; then
    echo "START_ID and END_ID must be non-negative integers" >&2
    exit 2
fi

if command -v scancel >/dev/null 2>&1; then
    :
else
    echo "scancel command not found in PATH" >&2
    exit 2
fi

if [ "$start" -le "$end" ]; then
    seq_cmd="$(seq "$start" "$end")"
else
    seq_cmd="$(seq "$start" -1 "$end")"
fi

for id in $seq_cmd; do
    echo "Cancelling job $id"
    if ! scancel "$id"; then
        echo "Failed to cancel job $id" >&2
    fi
done