CUDA_VISIBLE_DEVICES=0 \
python train_cknrm.py \
        -mode train \
        -train ../data/triples.train.small.tsv \
        -dev ../data/dev_toy.tsv \
        -qrels ../data/qrels_toy \
        -embed ../data/glove.6B.300d.txt \
        -vocab_size 400002 \
        -embed_dim 300 \
        -res ../results/cknrm.trec \
        -res_f ../results/cknrm_features \
        -depth 20 \
        -gamma 0.99 \
        -T 4 \
        -n_kernels 21 \
        -max_query_len 20 \
        -max_seq_len 128 \
        -epoch 1 \
        -batch_size 32
