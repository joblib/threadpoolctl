#!/bin/bash

set -e

python -m pip install --user codecov

python -m codecov || echo "codecov upload failed"
