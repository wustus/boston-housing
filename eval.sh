#!/bin/bash

echo "| Seed | Avg. Difference (10 Runs) |"
echo "| ---- | ------------------------- |"

TOTAL_AVG_DIFF=0

for seed in {42..51}; do 

    AVG_DIFF=0

    for _ in {1..10}; do
        RES=$(SEED=${seed} DEBUG=0 python3 src/main.py)
        DIFF=$(echo "${RES}" | awk -F: '{ print $2 }')
        AVG_DIFF=$(bc -e "${AVG_DIFF} + ${DIFF}")
    done

    AVG_DIFF=$(bc -e "scale=4; ${AVG_DIFF} / 10")
    TOTAL_AVG_DIFF=$(bc -e "${TOTAL_AVG_DIFF} + ${AVG_DIFF}")

    echo "| ${seed} | ${AVG_DIFF} |"
done

TOTAL_AVG_DIFF=$(bc -e "scale=4; ${TOTAL_AVG_DIFF} / 10")
echo "**AVG: ${TOTAL_AVG_DIFF}**"
