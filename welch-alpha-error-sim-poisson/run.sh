#!/bin/bash
set -e

sudo docker build -t welch-sim .
tmux new -s sim
sudo docker run -it --rm -v $(pwd)/output:/work/output welch-sim
sudo chown -R $(whoami):$(whoami) output/
aws s3 cp output/alpha_error_sim_poisson_result.parquet s3://welch-sim-test-komatsu-0112/alpha_error_sim_poisson_result.parquet

echo "完了"
