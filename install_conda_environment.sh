#!/usr/bin/env bash

set -e

ENVNAME='delhispatialindex'
PYTHONVERSION='3.13'

conda create -n $ENVNAME python=$PYTHONVERSION -y && \
    conda deactivate && \
    conda activate $ENVNAME && \
    conda install -c conda-forge poetry -y && \
    poetry install
