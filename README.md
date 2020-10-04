## Pytorch-Person-ReID-PCB-Beyond-Part-Models
* A Strong Implementation of PCB ([Beyond Part Models](https://arxiv.org/abs/1711.09349), ECCV2018) on Market-1501 and DukeMTMC-reID datasets.

* We support:
  * A strong PCB implementation which outperforms most existing implementations.
  * A simple and clear implementation, and end-to-end training and evaluation.


## News
* We re-write a strong Re-ID baseline [Bag of Tricks (BoT)](https://arxiv.org/abs/1903.07071) with a more simple and clear implementation, which is more friendly with researchers and newers. Our [code](https://github.com/wangguanan/Pytorch-Person-REID-Baseline-Bag-of-Tricks) can be found here. BoT outperforms PCB by using only the global feature. Our implementation of BoT achieves the same performance with the [offical one](https://github.com/michuanhaohao/reid-strong-baseline).

## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 0.4.0](http://pytorch.org/)
* GPU Memory >= 20G (we use 2 GTX1080ti)
* Memory >= 20G

## Dataset Preparation
* [Market-1501 Dataset](http://ww7.liangzheng.org/) and [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)
* Download and extract both anywhere

## Train and Test
```
python main.py --market_path market_path --duke_path duke_path
```

## Experiments

### 1. Settings
* We conduct our experiments on 2 GTX1080ti GPUs

### 2. Results

| Implementations | market2market | duke2duke | market2duke | duke2market |
| ---                               | :---: |:---: | :---: | :---: |
| PCB w/ REA (Ours) | **0.939 (0.832)** [<model.pth>](https://drive.google.com/file/d/1XzLE0nQZkLxzv0CJk-NMczCpHzCvegnM/view?usp=sharing) | 0.856 **(0.753)** [<model.pth>](https://drive.google.com/file/d/1AJ1n9YaO0lmHwWiV0g58f2WtPFvKFjjG/view?usp=sharing) | 0.384 (0.237) | 0.555 (0.285) | 
| PCB (Ours) | 0.934 (0.809) | **0.867** (0.746) | **0.440(0.265)** | **0.592 (0.308)** |
| PCB ([layumi](https://github.com/layumi/Person_reID_baseline_pytorch)) | 0.926 (0.774) | 0.642 (0.439) | - | - |
| PCB ([huanghoujing](https://github.com/huanghoujing/beyond-part-models)) | 0.928 (0.785) | 0.845 (0.700) | - | - |
| PCB ([Xiaoccer](https://github.com/Xiaoccer/ReID-PCB_RPP)) |	0.927 (0.796)	| - | - | - | 
| PCB (Paper) | 0.924 (0.773) | 0.819 (0.653)	| - | - |


## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
