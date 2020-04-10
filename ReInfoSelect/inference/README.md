# Inferece Code of Conv-KNRM on COVID19


## Datasets
* The input data (named ``bm25.jsonl``) is retrieved by BM25 method and each line of the file is constructed with below format
```
{"query_id": query_id, "query": query_text, "records": [{"paper_id": paper_id, "score": bm25 score, "paragraph": paragraph text}, ...]}
```
* Output is a jsonl file, the same format as input file.
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

