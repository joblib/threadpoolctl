#!/bin/bash

set -e

python -m pip --user install codecov

python -m codecov || echo "codecov upload failed"
