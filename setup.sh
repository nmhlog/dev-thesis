#!/bin/bash
apt-get update
apt install -y build-essential
apt-get install -y libboost-all-dev
apt-get install -y libsparsehash-dev
conda install -c bioconda google-sparsehash
conda install -y libboost
conda update -y libgcc
conda install -c conda-forge gcc
ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.6 /usr/lib/x86_64-linux-gnu/libmpfr.so.4
conda install -c daleydeng gcc-5
pip install -r requirements.txt
cd lib/spconv
LD_LIBRARY_PATH=/~/anaconda3/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
python setup.py bdist_wheel
cd dist
pip install "$(ls)"
cd ../..
cd hais_ops
export CPLUS_INCLUDE_PATH={conda_env_path}/hais/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
