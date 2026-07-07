#!/bin/bash
set -e

rm -rf output
sudo docker build -t welch-sim .
sudo docker run -it --rm -v $(pwd)/output:/work/output welch-sim
sudo chown -R $(whoami):$(whoami) output/
aws s3 cp output/ s3://welch-sim-test-komatsu-0112/ --recursive

echo "完了：S3にアップロード済み"
