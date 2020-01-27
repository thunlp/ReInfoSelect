import numpy as np

def ent_embloader(args):
    idx = 0
    idx2entity = []
    eneity2idx = {}
    entity2vec = {}

    # process unk pad
    idx2entity.append('<PAD>')
    entity2idx['<PAD>'] = idx
    entity2vec['<PAD>'] = np.random.normal(scale=0.6, size=(args.ent_embed_dim, ))
    idx += 1

    idx2entity.append('<UNK>')
    entity2idx['<UNK>'] = idx
    entity2vec['<UNK>'] = np.random.normal(scale=0.6, size=(args.ent_embed_dim, ))
    idx += 1

    with open(args.embed, 'r') as f:
        for line in f:
            val = line.split()
            idx2entity.append(val[0])
            entity2idx[val[0]] = idx
            entity2vec[val[0]] = np.asarray(val[1:], dtype='float32')
            idx += 1

    ent_embedding_init = np.zeros((len(idx2entity), args.ent_embed_dim))
    for idx, entity in enumerate(idx2entity):
        ent_embedding_init[idx] = entity2vec[entity]

    return entity2idx, ent_embedding_init
