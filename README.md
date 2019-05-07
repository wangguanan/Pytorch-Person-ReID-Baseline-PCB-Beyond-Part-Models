## Pytorch-PCB-Beyond-Part-Models
A Strong Implementation of PCB ([Beyond Part Models](https://arxiv.org/abs/1711.09349), ECCV2018) on Market-1501 and DukeMTMC-reID datasets.

## Dependencies
* [Anaconda (Python 2.7)](https://www.anaconda.com/download/)
* [PyTorch 0.4.0](http://pytorch.org/)
* GPU Memory >= 12G

## Dataset Preparation
* Download [Market-1501 Dataset](http://ww7.liangzheng.org/)
* Download [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)

## Train
```
python main.py --market_path market_path --duke_path duke_path
```

## Experiments
### Results
| Methods | market2market | duke2duke | market2duke | duke2market |

|s|

| Methods | market2market | duke2duke | market2duke | duke2market |	
|         | Rank-1 mAP | Rank-1 mAP | Rank-1 mAP | Rank-1 mAP |
| ---                               | :---: | :---: | :---: | :---: |
| PCB(Ours) | 0.928 0.797 | 0.859 0.737 | 0.459 0.276 | 0.556	0.287 |
| PCB([Zhedong Zheng](https://github.com/huanghoujing/beyond-part-models)) | 0.926	0.774	| 0.642	0.439 |	 |  |
| PCB([Houjing Huang](https://github.com/huanghoujing/beyond-part-models)) |	0.928	0.785	| 0.845	0.7	|   |  |
| PCB([Xiaoccer](https://github.com/Xiaoccer/ReID-PCB_RPP)) |	0.927	0.786	|  |  |  | 
| PCB(Paper) | 0.924 0.773 | 0.819	0.653	|  |  |


* Market-1501: Rank-1 0.928, mAP 0.797
* DukeMTMC-reID: Rank1 0.859, mAP 0.737
* More results and parameter analysis are coming soon

## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
