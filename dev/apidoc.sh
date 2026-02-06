#!/bin/bash
# Generate API documentation, overwriting previous version, should be run from repository root

rm -rf docs/source/api/
sphinx-apidoc --no-toc -M -f --implicit-namespaces -o docs/source/api/ tomotok/ tomotok/*/tests/*

rm docs/source/api/tomotok.rst
