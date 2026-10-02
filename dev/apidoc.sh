#!/bin/bash
# Generate API documentation, overwriting previous version, should be run from repository root

rm -rf docs/source/api/
# tomotok/core is a deprecated compatibility layer and is not documented
sphinx-apidoc --no-toc -M -f --implicit-namespaces -o docs/source/api/ tomotok/ tomotok/core

rm docs/source/api/tomotok.rst
