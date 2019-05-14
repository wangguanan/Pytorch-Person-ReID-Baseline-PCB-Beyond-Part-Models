## Pytorch-Person-ReID-PCB-Beyond-Part-Models
* A Strong Implementation of PCB ([Beyond Part Models](https://arxiv.org/abs/1711.09349), ECCV2018) on Market-1501 and DukeMTMC-reID datasets.

## Dependencies
* [Anaconda (Python 2.7)](https://www.anaconda.com/download/)
* [PyTorch 0.4.0](http://pytorch.org/)
* GPU Memory >= 12G
* Memory >= 20G

## Dataset Preparation
* [Market-1501 Dataset](http://ww7.liangzheng.org/) and [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)
* Download and extract both anywhere

## Train and Test
```
python main.py --market_path market_path --duke_path duke_path
```

## Experiments
### Results

| Implementations | market2market | duke2duke | market2duke | duke2market |
| ---                               | :---: | :---: | :---: | :---: |
| PCB (Ours) | **0.938 (0.803)** | **0.864 (0.741)** | **0.446 (0.264)** | **0.579 (0.296)** |
| PCB ([layumi](https://github.com/layumi/Person_reID_baseline_pytorch)) | 0.926 (0.774) | 0.642 (0.439) | - | - |
| PCB ([huanghoujing](https://github.com/huanghoujing/beyond-part-models)) | 0.928 (0.785) | 0.845 (0.700) | - | - |
| PCB ([Xiaoccer](https://github.com/Xiaoccer/ReID-PCB_RPP)) |	0.927 (0.796)	| - | - | - | 
| PCB (Paper) | 0.924 (0.773) | 0.819 (0.653)	| - | - |

* More results and parameter analysis are coming soon

## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
