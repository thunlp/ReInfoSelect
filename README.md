# ReInfoSelect
Code and dataset of WWW2020 Paper **Selective Weak Supervision for Neural Information Retrieval**. [Paper](https://arxiv.org/pdf/2001.10382.pdf)

![ReInfoSelect](ReInfoSelect.png)

## Results
|Method|cw09 NDCG@20|cw09 ERR@20|rb04 NDCG@20|rb04 ERR@20|cw12 NDCG@20|cw12 ERR@20|
|:----:|:----------:|:---------:|:----------:|:---------:|:----------:|:---------:|
|Conv-KNRM|||||
|No Weak Supervision|0.2873|0.1597|0.4267|0.1168|0.1123|0.0915|
|Anchor+BM25 Labels|0.2910|0.1585|0.4322|0.1179|0.1181|0.0978|
|Title Discriminator|0.2927|0.1606|0.4318|0.1193|0.1176|0.0975|
|All Anchor|0.2839|0.1464|0.4305|0.1190|0.1119|0.0906|
|MS MARCO Human Label|0.2903|0.1542|0.4337|0.1194|0.1183|0.0981|
|ReInfoSelect|0.3096|0.1611|0.4423|0.1202|0.1225|0.1044|
|ReInfoSelect (ensemble)|0.3244|0.1778|0.4503|0.1227|0.1279|0.1042|
|BERT|||||||
|No Weak Supervision|0.2999|0.1631|0.4258|0.1163|0.1190|0.0963|
|Anchor+BM25 Labels|0.3068|0.1618|0.4375|0.1233|0.1160|0.0990|
|Title Discriminator|0.3021|0.1513|0.4379|0.1202|0.1162|0.0981|
|All Anchor|0.3072|0.1609|0.4446|0.1206|0.1208|0.0965|
|MS MARCO Human Label|0.3085|0.1652|0.4415|0.1213|0.1207|0.1024|
|ReInfoSelect|0.3261|0.1669|0.4500|0.1220|0.1276|0.0997|
|ReInfoSelect (ensemble)|0.3391|0.1815|0.4613|0.1287|0.1302|0.1038|
|EDRM|||||
|No Weak Supervision|0.2922|0.1642|0.4263|0.1158|0.1119|0.0910|
|Anchor+BM25 Labels|0.2989|0.1650|0.4341|0.1179|0.1172|0.0947|
|Title Discriminator|0.2983|0.1642|0.4315|0.1167|0.1176|0.0950|
|All Anchor|0.3012|0.1715|0.4311|0.1175|0.1167|0.0958|
|ReInfoSelect|0.3163|0.1794|0.4396|0.1208|0.1215|0.0980|
|TK|||||
|No Weak Supervision|0.3003|0.1577|0.4273|0.1163|0.1192|0.0991|
|ReInfoSelect|0.3103|0.1626|0.4320|0.1183|0.1297|0.1043|

## Datasets
Coming soon.

## References
- [End-to-end neural ad-hoc ranking with kernel pooling](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf) (K-NRM)
- [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf) (Conv-KNRM)
- [TU Wien @ TREC Deep Learning '19 - Simple Contextualization for Re-ranking](https://arxiv.org/pdf/1912.01385.pdf) (TK)
- [Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval](https://www.aclweb.org/anthology/P18-1223.pdf) (EDRM)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) (BERT)

## Contact
If you have questions, suggestions and bug reports, please email zkt18@mails.tsinghua.edu.cn.
