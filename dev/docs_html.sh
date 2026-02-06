#!/bin/bash
# Generate HTML documentation, should be run from repository root

DOCDIR=docs
BUILDDIR=$DOCDIR/build

sphinx-build -M html -d $BUILDDIR/doctrees $DOCDIR/source $BUILDDIR/html
