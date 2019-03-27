#!/bin/bash

set -e

pip install codecov

codecov || echo "codecov upload failed"
