#!/usr/bin/env bash
# Reads changed file names on stdin; exits 0 when one matches a line of slow-models-paths.txt.
exec grep -qF -f "$(dirname "$0")/../slow-models-paths.txt"
