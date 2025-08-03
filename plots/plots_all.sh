#!/bin/bash

find . -type d | while read dir; do
    if ls "$dir"/*.py >/dev/null 2>&1; then
        echo "Enter: $dir"
        cd "$dir" || continue

        for pyfile in *.py; do
            echo "Execute: $pyfile"
            python "$pyfile"
        done
        cd - >/dev/null
    fi
done
