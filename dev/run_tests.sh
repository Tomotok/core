#!/bin/bash
# Run the full test suite, should be run from repository root

python3 -m unittest discover -s tests -t . -p "test_*.py"
