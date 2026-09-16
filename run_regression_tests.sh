#!/bin/bash

pip uninstall -y RL4CC
pip install .

echo
echo "########################################################################"
echo

cd RL4CC && python run_regression_tests.py && cd ..
