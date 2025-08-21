@echo off
call conda create --name pytorch36 python=3.6 -y 
call conda activate pytorch36
call conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
call conda install -y -c intel scikit-learn
call conda install -y -c conda-forge numpy
call conda install -y -c conda-forge pandas
call conda install -y -c conda-forge  pathlib
call conda install -y -c conda-forge  pydicom
call conda install -y -c conda-forge  tqdm
call conda install -y -c conda-forge scikit-image
call conda install -y -c conda-forge time
call conda install -y -c conda-forge matplotlib
call conda install -y -c conda-forge scipy
call conda install -y -c conda-forge pytest
call conda install -y -c conda-forge gdcm
call conda install -y jupyter
call conda deactivate
