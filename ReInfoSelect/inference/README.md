# Inferece Code of Conv-KNRM on COVID19


## Datasets
* The input data (named ``test.txt``) is retrieved by BM25 method and the file is constructed with below format
```
query text \t document text \t relevance label \t query id \t document id \t bm25 score  
```
* Output is a trec file
* All resouce can be found at Amazon Web Services.
```
https://thunlp.s3-us-west-1.amazonaws.com/reinfoselect_cknrm_covid19
```

## Requirements

* `python > 3.0`
* `torch >= 1.0.0`
* `nltk`
* `krovetzstemmer`

To run the inference code.

```
bash ./inference.sh
```

