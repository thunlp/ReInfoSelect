CUDA_VISIBLE_DEVICES=0 python cknrm_inference.py \
--test_file ./dlg_bm25.jsonl \
--out_path ./dlg_marco_recknrm.jsonl \
--pretrained_model ./dlg_marco_recknrm_128 \
--embedding_path ./glove.6B.300d.txt \
--max_doc_len 128
