### 0. Thanks
We express our highest respect and gratitude for the open-source work of [BasicSR](https://github.com/XPixelGroup/BasicSR).
### 1. Environment preparation
Anaconda is suggested. Install the necessary packages:
```shell
conda create -n lqct_sr_dn python=3.9.7
conda activate lqct_sr_dn
pip install -r requirements.txt
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -e .
```
### 2. Data preparation
You should prepare your data in this way:
```
data_rootdir
    - dataset_name
        - img
            - hr_nd
                - train
                - val
                - test
            - lr_ld
                - x2
                    - train
                    - train_avg
                    - val
                    - val_avg
                    - test
                    - test_avg
                - x4
                    - train
                    - train_avg
                    - val
                    - val_avg
                    - test
                    - test_avg
            - lr_nd
                - x2
                    - train
                    - val
                    - test
                - x4
                    - train
                    - val
                    - test
        -mask
            - hr
                - train
                - val
                - test
            - x2
                - train
                - val
                - test
            - x4
                - train
                - val
                - test
```
### 3. Train
To train the network, you should modify the config files in "options/train" folder first.

Train the network with the scale factor of 2:
```shell
python basicsr/train.py -opt options/train/sr_dn_x2.yml
```
Train the network with the scale factor of 4:
```shell
python basicsr/train.py -opt options/train/sr_dn_x4.yml
```
### 4. Test
To test the network, you should modify the config files in "options/test" folder first.

Test the network with the scale factor of 2:
```shell
python basicsr/train.py -opt options/test/sr_dn_x2.yml
```
Test the network with the scale factor of 4:
```shell
python basicsr/train.py -opt options/train/sr_dn_x4.yml
```