#!/bin/bash
set -e

if [ -d "t-test" ]; then
  cd t-test
  git pull
  cd welch-alpha-error-sim
else
  git clone https://github.com/Komatsu-T/t-test.git
  cd t-test/welch-alpha-error-sim
fi

sudo docker build -t welch-sim .
sudo docker run -it --rm -v $(pwd)/output:/work/output welch-sim
sudo chown -R $(whoami):$(whoami) output/

echo "完了"