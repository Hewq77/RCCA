# RCCA 
This repository contains the PyTorch implementation of the paper "A New Context-Aware Framework for Defending Against Adversarial Attacks in Hyperspectral Image Classification". (_IEEE TGRS 2023_) [[pdf]](https://ieeexplore.ieee.org/document/10056357)  
By [Bing Tu](https://faculty.nuist.edu.cn/tubing/zh_CN/index/144219/list/index.htm), Wangquan He, Qianming Li, Yishu Peng, and [Antonio Plaza](https://www2.umbc.edu/rssipl/people/aplaza/)


## Usage
### 1. Installation
Requirements are Python 3.7 and PyTorch 1.4.0.

### 2. Download data
Download the dataset from [[here]](https://picture.iczhiku.com/weixin/message1590686900389.html), and put it in the 'Data' directory.

### 3. Training and test
To train the model with different attack strategies, please run the following command:
```sh
python demo_RCCA_FGSM.py --attack FGSM
```

In addition, the dataset and model can be replaced by changing `--dataID` and `--model` in 'demo_RCCA_FGSM.py'.
    
## Citation
Please consider citing our work if you think it is useful for your research：
```
@ARTICLE{tu2023new,
  author={Tu, Bing and He, Wangquan and Li, Qianming and Peng, Yishu and Plaza, Antonio},
  journal={IEEE Trans. Geosci. Remote Sens.}, 
  title={A New Context-Aware Framework for Defending Against Adversarial Attacks in Hyperspectral Image Classification}, 
  year={2023},
  volume={61},
  month={Feb},
  pages={1-14},
  doi={10.1109/TGRS.2023.3250450}}
```

**Acknowledgment**: This code is based on the [SACNet](https://github.com/YonghaoXu/SACNet?tab=readme-ov-file) and [UAE-RS](https://github.com/YonghaoXu/UAE-RS). Thanks to the authors for their wonderful work.

