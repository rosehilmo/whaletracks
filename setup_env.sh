#! /bin/bash
# Sets up the conda environment whaletracks
NAME="whaletracks"
# Install LFS
# https://github.com/git-lfs/git-lfs/wiki/Installation
brew install curl
#curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
brew install git-lfs
git lfs install
git lfs track *.db
git lfs track *.csv
# Setup for project
conda config --set always_yes yes --set changeps1 no
conda update --quiet conda
conda info --all
conda env create --quiet --file environment.yml
conda install jupyter notebook
# Install the database browser
brew install sqlitebrowser
echo "Use 'conda activate ${NAME}' to enter environment."
echo "Use 'conda deactivate' to exit environment."
