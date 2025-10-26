# Installation

```
git clone git@github.com:ChengmingZhang-CAS/CauTrigger-main.git
cd CauTrigger-main
conda create -n CauTrigger python==3.10
conda activate CauTrigger
pip install -r requirements.txt

# install torch
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

python setup.py install
```

