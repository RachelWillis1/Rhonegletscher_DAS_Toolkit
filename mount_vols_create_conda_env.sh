#!/bin/bash

# Format volumes as an ext4 filesystem (or another format you prefer)
mkfs -t ext4 /dev/nvme1n1
mkfs -t ext4 /dev/nvme2n1

# Mount Volumes to directories
mkdir Anaconda
mkdir Rhone_Glacier

mount /dev/nvme1n1 Anaconda
mount /dev/nvme2n1 Rhone_Glacier

# Download Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
chmod +x Anaconda3-2020.02-Linux-x86_64.sh
./Anaconda3-2020.02-Linux-x86_64.sh

# Activate conda
source /home/ec2-user/Anaconda/anaconda3/bin/activate

# Create conda environment
conda create --name py3 python=3.9

# Activate conda environment
conda activate py3

# Install dependencies

# Download dependencies
conda install numpy==1.26.4
conda install matplotlib==3.9.2
conda install pandas==2.2.3
conda install scipy==1.12.0
conda install h5py==3.12.1
conda install scikit-learn==1.1.2
pip install obspy==1.4.1
pip install pyasdf==0.8.1
pip install dask
pip install "dask[distributed]" --upgrade
pip install joblib==1.2.0
