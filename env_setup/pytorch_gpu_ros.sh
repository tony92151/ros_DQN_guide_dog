echo "Install pytorch for ros"

echo "PRESS [ENTER] TO CONTINUE THE INSTALLATION"
echo "IF YOU WANT TO CANCEL, PRESS [CTRL] + [C]"
read


echo "[Install cuda9.0 & cudnn7.3]"
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo dpkg -i libcudnn7_7.3.1.20-1+cuda9.0_amd64.deb
source ~/.bashrc




echo "[Setup env, use pip]"
sudo apt-get install python-virtualenv
sudo apt install cmake

sh -c "echo \"alias tros='source ~/torch_gpu_ros/bin/activate'\" >> ~/.bashrc"

source $HOME/.bashrc

cd

virtualenv --system-site-packages -p python3 ~/torch_gpu_ros
source ~/torch_gpu_ros/bin/activate


echo "[Download and install pytorch]"
pip install torch torchvision
pip install numpy
pip install -U rosinstall msgpack empy defusedxml netifaces


echo "pytorch version"
python -c "import torch; print(torch.__version__)"

echo "Everytime you want to get into virtualenv of pytorch, just typing [ tros ]"
echo "Install Finish!!!"

