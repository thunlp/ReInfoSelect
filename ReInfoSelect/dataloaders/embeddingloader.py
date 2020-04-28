import numpy as np

def embeddingloader(args):
    idx = 0
    idx2word = []
    word2idx = {}
    word2vec = {}

    # process unk pad
    idx2word.append('<PAD>')
    word2idx['<PAD>'] = idx
    word2vec['<PAD>'] = np.random.normal(scale=0.6, size=(300, ))
    idx += 1

    idx2word.append('<UNK>')
    word2idx['<UNK>'] = idx
    word2vec['<UNK>'] = np.random.normal(scale=0.6, size=(300, ))
    idx += 1

    with open(args.embed, 'r') as f:
        for line in f:
            val = line.split()
            idx2word.append(val[0])
            word2idx[val[0]] = idx
            word2vec[val[0]] = np.asarray(val[1:], dtype='float32')
            idx += 1

    embedding_init = np.zeros((len(idx2word), args.embed_dim))
    for idx, word in enumerate(idx2word):
        embedding_init[idx] = word2vec[word]

    return embedding_init
