echo [$(date)]: "START" 
echo [$(date)]: "creating env with python 3.8 version" 
conda create -n cifar100 python=3.8 -y
echo [$(date)]: "activating the environment" 
conda activate cifar100
echo [$(date)]: "installing the dev requirements" 
pip install -r requirements_dev.txt
echo [$(date)]: "END" 