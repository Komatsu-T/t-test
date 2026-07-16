#!/bin/bash
set -e

rm -rf output
docker build -t shash-sim .
docker run --rm -v $(pwd)/output:/work/output shash-sim

echo "完了"
