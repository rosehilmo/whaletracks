#!/bin/bash
# Setup python paths to run codes
PYTHONPATH=${PYTHONPATH}:`pwd`
export PYTHONPATH
cd common_python
PYTHONPATH=${PYTHONPATH}:`pwd`
cd ..
