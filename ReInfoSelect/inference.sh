CUDA_VISIBLE_DEVICES=0 \
python main.py \
        -mode infer \
        -model cknrm \
        -max_input 1280000 \
        -checkpoint ../checkpoints/reinfoselect_cknrm.bin \
        -dev ../data/dev_toy.tsv \
        -embed ../data/glove.6B.300d.txt \
        -vocab_size 400002 \
        -embed_dim 300 \
        -res_trec ../results/cknrm.trec \
        -res_json ../results/cknrm.json \
        -res_feature ../features/cknrm_features \
        -gamma 0.99 \
        -T 1 \
        -n_kernels 21 \
        -max_query_len 20 \
        -max_seq_len 128 \
        -epoch 1 \
        -batch_size 32
