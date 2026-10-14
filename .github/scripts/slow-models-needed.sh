#!/usr/bin/env bash
# Reads changed file names on stdin. Exits 0 when one starts with a line of slow-models-paths.txt,
# 1 when none does and 2 when the list cannot be read.
paths="$(dirname "$0")/../slow-models-paths.txt"
[ -r "$paths" ] || exit 2
mapfile -t prefixes < <(grep -v '^[[:space:]]*$' "$paths")
while IFS= read -r changed; do
  for prefix in "${prefixes[@]}"; do
    [[ "$changed" == "$prefix"* ]] && exit 0
  done
done
exit 1
