CUDA_VISIBLE_DEVICES=0 python cknrm_inference.py \
--test_file ./bm25.jsonl \
--out_path ./cknrm.jsonl \
--pretrained_model ./reinfoselect_cknrm_covid19 \
--embedding_path ./glove.6B.300d.txt
