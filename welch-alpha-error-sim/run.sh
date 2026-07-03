#!/bin/bash
set -e

sudo docker build -t welch-sim .
sudo docker run -it --rm -v $(pwd)/output:/work/output welch-sim
sudo chown -R $(whoami):$(whoami) output/

echo "完了"
