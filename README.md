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
  <table width="492" border="0" cellpadding="0" cellspacing="0" style='width:492.00pt;border-collapse:collapse;table-layout:fixed;'>
   <col width="87.20" style='mso-width-source:userset;mso-width-alt:3720;'/>
   <col width="74.40" span="2" style='mso-width-source:userset;mso-width-alt:3174;'/>
   <col width="51.20" span="5" style='width:51.20pt;'/>
   <tr height="17.60" style='height:17.60pt;'>
    <td class="xl65" height="17.60" width="492" colspan="8" style='height:17.60pt;width:492.00pt;border-right:none;border-bottom:none;' x:str>source2target</td>
   </tr>
   <tr height="17.60" style='height:17.60pt;'>
    <td class="xl65" height="17.60" colspan="2" style='height:17.60pt;border-right:none;border-bottom:none;' x:str>market2market</td>
    <td class="xl65" colspan="2" style='border-right:none;border-bottom:none;' x:str>market2duke</td>
    <td class="xl65" colspan="2" style='border-right:none;border-bottom:none;' x:str>duke2duke</td>
    <td class="xl65" colspan="2" style='border-right:none;border-bottom:none;' x:str>duke2market</td>
   </tr>
   <tr height="17.60" style='height:17.60pt;'>
    <td height="17.60" style='height:17.60pt;' x:str>Rank-1</td>
    <td x:str>mAP</td>
    <td x:str>Rank-1</td>
    <td x:str>mAP</td>
    <td x:str>Rank-1</td>
    <td x:str>mAP</td>
    <td x:str>Rank-1</td>
    <td x:str>mAP</td>
   </tr>
   <![if supportMisalignedColumns]>
    <tr width="0" style='display:none;'>
     <td width="87" style='width:87;'></td>
     <td width="74" style='width:74;'></td>
    </tr>
   <![endif]>
  </table>

* Market-1501: Rank-1 0.928, mAP 0.797
* DukeMTMC-reID: Rank1 0.859, mAP 0.737
* More results and parameter analysis are coming soon

## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
