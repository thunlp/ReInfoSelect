# ReInfoSelect
Source Code of Selective Weak Supervision for Neural Information Retrieval.

### Neural Ranking Models
#### BERT
The BERT model is corresponding to `/models/bert.py`. This Model is based on the inplementation of huggingface's [transformers](https://github.com/huggingface/transformers), and is initialized with [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin).

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

#### EDRM
The EDRM model is corresponding to `/models/edrm.py`.

- [Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval](https://www.aclweb.org/anthology/P18-1223.pdf)

#### TK
The TK model is corresponding to `/models/tk.py`.

- [TU Wien @ TREC Deep Learning ’19 – Simple Contextualization for Re-ranking](https://arxiv.org/pdf/1912.01385.pdf)

#### Conv-KNRM
The Conv-KNRM model is corresponding to `/models/cknrm.py`.

- [Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search](http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)

#### KNRM
The KNRM model is corresponding to `/models/knrm.py`.

- [End-to-end neural ad-hoc ranking with kernel pooling](http://www.cs.cmu.edu/afs/cs/user/cx/www/papers/K-NRM.pdf)

### Data Selection Policies
#### Query Policy
The query policy is corresponding to `/policies/q_policy.py`.

#### Query-Document Interaction Policy
The query-document interaction policy is corresponding to `/policies/qd_policy.py`.

#### Query, Document and Interaction Policy
The query document and interaction policy is corresponding to `/policies/all_policy.py`.

## Contact
If you have questions, suggestions and bug reports, please email zkt18@mails.tsinghua.edu.cn.
