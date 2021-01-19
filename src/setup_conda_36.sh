#!/bin/bash
# note that anaconda was already installed on my workstation at time of writing this script. This is intended to install anaconda and create the required environment in an ubuntu OS environment
set -e
try_conda=$(which conda | wc -l)

if [ $try_conda -lt 1 ];
then
	curl https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh > ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh
	bash ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh

fi


# create environment if does not exist
has_pytorch_env=$(conda env list | grep -E  'envs/pytorch36$' | wc -l)

if [ $has_pytorch_env -lt 1 ];
then
	conda create --name pytorch36
fi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pytorch36
conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -y -c intel scikit-learn
conda install -y -c conda-forge numpy
conda install -y -c conda-forge pandas
conda install -y -c conda-forge  pathlib
conda install -y -c conda-forge  pydicom
conda install -y -c conda-forge  tqdm
conda install -y -c conda-forge scikit-image
conda install -y -c conda-forge time
conda install -y -c conda-forge matplotlib
conda install -y -c conda-forge scipy
conda install -y -c conda-forge pytest
conda install -y -c conda-forge gdcm
conda install -y -c anaconda memory_profiler
# note that for nbconvert you need XeLaTeX. since this requires sudo permissions, do this command manually after running this script
# sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-generic-recommended
conda install -y jupyter
conda deactivate
