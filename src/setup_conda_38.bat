@echo off
call conda create --name pytorch38 python=3.8 -y 
call conda activate pytorch38
call conda install -y seaborn
call conda install -y pytorch torchvision cudatoolkit=10.2 -c pytorch
call conda install -y -c intel scikit-learn
call conda install -y -c conda-forge numpy
call conda install -y -c conda-forge pandas
call conda install -y -c conda-forge scikit-image
call conda install -y -c conda-forge time
call conda install -y -c conda-forge matplotlib
call conda install -y -c conda-forge scipy
call conda install -y -c conda-forge pytest
call conda install -y jupyter
call conda install -y psutil
call conda install -y tqdm
call conda deactivate
