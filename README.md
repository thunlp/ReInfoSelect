# ReInfoSelect
Codes and datasets of WWW2020 Paper **Selective Weak Supervision for Neural Information Retrieval**. [Paper](https://arxiv.org/pdf/2001.10382.pdf)

## Framework

![ReInfoSelect](./framework/ReInfoSelect.png)

## Results

|Model|Method|ClueWeb09 NDCG@20|ClueWeb09 ERR@20|Robust04 NDCG@20|Robust04 ERR@20|ClueWeb12 NDCG@20|ClueWeb12 ERR@20|
|:----|:----:|:---------------:|:--------------:|:--------------:|:-------------:|:---------------:|:--------------:|
|[Conv-KNRM](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)|**No Weak Supervision**|0.2873|0.1597|0.4267|0.1168|0.1123|0.0915|
||**ReInfoSelect**|0.3096|0.1611|0.4423|0.1202|0.1225|***0.1044***|
|[TK](https://arxiv.org/pdf/1912.01385.pdf)|**No Weak Supervision**|0.3003|0.1577|0.4273|0.1163|0.1192|0.0991|
||**ReInfoSelect**|0.3103|0.1626|0.4320|0.1183|***0.1297***|0.1043|
|[EDRM](https://www.aclweb.org/anthology/P18-1223.pdf)|**No Weak Supervision**|0.2922|0.1642|0.4263|0.1158|0.1119|0.0910|
||**ReInfoSelect**|0.3163|***0.1794***|0.4396|0.1208|0.1215|0.0980|
|[BERT](https://arxiv.org/pdf/1810.04805.pdf)|**No Weak Supervision**|0.2999|0.1631|0.4258|0.1163|0.1190|0.0963|
||**ReInfoSelect**|***0.3261***|0.1669|***0.4500***|***0.1220***|0.1276|0.0997|

More results are available in [results](./results).

## Datasets
Data can be downloaded from [Datasets](https://cloud.tsinghua.edu.cn/d/77741ef1c1704866814a/).

|Datasets|Queries/Anchors|Query/Anchor-Doc Pairs|Released Files|
|:-------|:-------------:|:--------------------:|:-------------|
|**Weak Supervision**|100K|6.75M|Anchors, A-D Relations|
|**ClueWeb09-B**|200|47.1K|Queries, Q-D Relations, SDM scores|
|**Robust04**|249|311K|Queries, Q-D Relations, SDM scores|
|**ClueWeb12-B13**|100|28.9K|Queries, Q-D Relations, SDM scores|

As we cannot release the document contents, the document IDs are used instead.

## Requirements

* `python == 3.6` or `3.7`
* `torch >= 1.0.0`.

To run ReInfoSelect, please install all requirements.
```
pip install -r requirements.txt
```

## Get Started
First, please prepare your data in recommended [format](./data).
```
python ./data/preprocess.py
```

For Conv-KNRM,
```
bash ./ReInfoSelect/train_cknrm.sh
```

and for BERT,
```
bash ./ReInfoSelect/train_bert.sh
```

After ReInfoSelect training, please concatenate the neural features with SDM score. Then feed all features into Coor-Ascent using [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/).

## Citation
Please cite our paper if you find it helpful.

```
@inproceedings{zhang2020selective,
    title = {Selective Weak Supervision for Neural Information Retrieval},
    author = {Kaitao Zhang and Chenyan Xiong and Zhenghao Liu and Zhiyuan Liu},
    booktitle = {Proceedings of WWW},
    year = {2020}
}
```

## Contact
If you have any questions, suggestions and bug reports, please email zkt18@mails.tsinghua.edu.cn.
