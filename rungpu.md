# CUDA toolkit for WSL (never install a Linux GPU driver here; Windows' driver is shared)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# QES's libraries (the list from QES's README)
sudo apt-get -y install build-essential git libgdal-dev libnetcdf-dev libnetcdf-c++4-dev libnetcdf-cxx-legacy-dev netcdf-bin libboost-all-dev cmake

# your repo, on the fitfloat branch
git clone -b fitfloat https://github.com/Juliatduffy/QES-benchmarking.git ~/QES

# the two fire files, straight from CHPC (asks for your password + Duo)
scp u1417379@granite.chpc.utah.edu:QES/QES-Public/data/FireFiles/{HeatPot.nc,FFII.nc} ~/QES/QES-Public/data/FireFiles/

#set up vm
cd ~
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
tar -xf vscode_cli.tar.gz
./code tunnel
