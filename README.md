# ReInfoSelect
Codes and datasets of WWW2020 Paper **Selective Weak Supervision for Neural Information Retrieval**. [Paper](https://arxiv.org/pdf/2001.10382.pdf)

![ReInfoSelect](ReInfoSelect.png)

## Datasets
Data can be downloaded from [Data](https://cloud.tsinghua.edu.cn/d/77741ef1c1704866814a/)

Datasets include queries, qrels and SDM rankings (.trec) for ClueWeb09-B, Robust04 and ClueWeb12-B13. We also release the weak supervision relation, all anchor and 100K anchor files. However, We cannot release the document contents.

## Requirements
This repository has been tested with `Python 3.7` and `pytorch>=1.0`.

Other requirements include `texar_pytorch 0.1.1`, `allennlp 0.9.0`, `transformers 2.4.1`, `krovetzstemmer 0.6`, `nltk 3.4.5` and `pytrec_eval 0.4`.

## Get Started
First, please prepare the needed data in recommended format.

For kernel-based neural ranking models,
```
python train_kernel.py
```

and for BERT,
```
python train_bert.py
```

## References
- [End-to-end neural ad-hoc ranking with kernel pooling](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf) (K-NRM)
- [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) (Conv-KNRM)
- [TU Wien @ TREC Deep Learning '19 - Simple Contextualization for Re-ranking](https://arxiv.org/pdf/1912.01385.pdf) (TK)
- [Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval](https://www.aclweb.org/anthology/P18-1223.pdf) (EDRM)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (BERT)

## Citation

## Contact
If you have questions, suggestions and bug reports, please email zkt18@mails.tsinghua.edu.cn.
