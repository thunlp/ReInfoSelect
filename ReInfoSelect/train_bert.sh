CUDA_VISIBLE_DEVICES=0 \
python train_bert.py \
        -train ../data/weak_supervision.tsv \
        -dev ../data/dev.tsv \
        -qrels ../data/qrels \
        -embed ../data/glove.6B.300d.txt \
        -model bert-base-uncased \
        -vocab_size 400002 \
        -embed_dim 300 \
        -res ../results/bert_out.trec \
        -depth 20 \
        -gamma 0.99 \
        -T 4 \
        -n_kernels 21 \
        -max_query_len 20 \
        -max_seq_len 150 \
        -epoch 1 \
        -batch_size 4
