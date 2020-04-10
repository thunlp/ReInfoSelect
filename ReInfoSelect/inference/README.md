# Inferece Code of Conv-KNRM on COVID19

## Datasets
* The input data (named ``bm25.jsonl``) is retrieved by BM25 method and each line of the file is constructed with below format
```
{"query_id": query_id, "query": query_text, "records": [{"paper_id": paper_id, "score": bm25 score, "paragraph": paragraph text}, ...]}
```
* Output is a jsonl file, the same format as input file.
* Checkpoints are available at Amazon Web Services.
```
https://thunlp.s3-us-west-1.amazonaws.com/reinfoselect_cknrm_covid19
```

## Requirements

* `python == 3.7`
* `torch >= 1.0.0`

Please install all requirements.
```
pip install -r ../../requirements.txt
```

To run the inference code.

```
bash ./inference.sh
```
or
```
python cknrm_inference.py \
--test_file {input jsonl} \
--out_path {output jsonl} \
--pretrained_model {cknrm checkpoint} \
--embedding_path {glove.6B.300d.txt}
```
