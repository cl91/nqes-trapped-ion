#!/bin/bash

MAX_TRIES=16
for i in $(seq $MAX_TRIES); do
    $@ --throw-fperror
    exit_code=$?
    echo "Program exited with code $exit_code"
    if [[ "$exit_code" -ne 127 ]]; then
	if [[ "$exit_code" -eq 0 ]]; then
	    echo "========"
	    echo "Success!"
	    echo "========"
	fi
	exit $exit_code
    fi
    echo "====================================="
    echo "Restarting computation with new seed."
    echo "====================================="
done
