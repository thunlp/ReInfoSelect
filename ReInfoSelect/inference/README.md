# Inferece Code of Conv-KNRM on COVID19

## Datasets
* The input data (named ``dlg_bm25.jsonl``) is retrieved by BM25 method and each line of the file is constructed with below format
```
{"query_id": query_id, "query": query_text, "records": [{"paper_id": paper_id, "score": bm25 score, "paragraph": paragraph text}, ...]}
```
* Output is a jsonl file, the same format as input file.
* Get glove embeddings.
```
wget http://nlp.stanford.edu/data/glove.6B.zip
```
* Get checkpoints.
```
wget https://thunlp.s3-us-west-1.amazonaws.com/dlg_marco_recknrm_128
wget https://thunlp.s3-us-west-1.amazonaws.com/dlg_marco_recknrm_256
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
CUDA_VISIBLE_DEVICES=0 python cknrm_inference.py \
--test_file {path to input jsonl} \
--out_path {path to output jsonl} \
--pretrained_model {path to a single checkpoint file or a folder which contains several checkpoints} \
--embedding_path {path to glove.6B.300d.txt} \
--max_doc_len {max_doc_len}
```
