import re
import json
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from nltk.corpus import stopwords
sws = {}
for w in stopwords.words('english'):
    sws[w] = 1
from krovetzstemmer import Stemmer
stemmer = Stemmer()



test_file = "/home3/zhangkaitao/retrieval/test_cx/dev_ext.tsv"
pretrained_model = '/home1/zhangkaitao/models/reinfoselect_cknrm_kt4'

regex_drop_char = re.compile('[^a-z0-9\s]+')
regex_multi_space = re.compile('\s+')

def kernal_mus(n_kernels):
    """
    get the mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each gaussian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma


class cknrm(nn.Module):
    """
    kernel pooling layer
    """

    def __init__(self, num_kernels, rembed_dim, cnn_kernel, ifcuda, lock):
        """

        :param mu: |d| * 1 dimension mu
        :param sigma: |d| * 1 dimension sigma
        """
        super(cknrm, self).__init__()
        tensor_mu = torch.FloatTensor(kernal_mus(num_kernels))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(num_kernels))
        if ifcuda and torch.cuda.is_available():
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        '''
        if opt.cuda:
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        '''

        self.d_word_vec = rembed_dim
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, num_kernels)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, num_kernels)
        self.tanh = nn.Tanh()
        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, cnn_kernel, (1, rembed_dim)),
            nn.ReLU()
        )

        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, cnn_kernel, (2, rembed_dim)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, cnn_kernel, (3, rembed_dim)),
            nn.ReLU()
        )
        if lock:  # toby add
            for p in self.parameters():
                p.requires_grad = False
        self.dense_f = nn.Linear(num_kernels * 9, 1, 1)
        self.dense_fpp = nn.Linear(num_kernels * 9 + 1, 1, 1)  # toby add

        torch.nn.init.uniform_(self.dense_f.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_fpp.weight, -0.014, 0.014)  # inits taken from matchzoo

    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        # np.save('/data/disk4/private/zhangjuexiao/e2e_case/rank_case'+"_"+str(q_embed.size()[1])+"_"+str(d_embed.size()[2]), sim.data.cpu().numpy())
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * atten_q.type_as(self.sigma)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def forward(self, qw_embed, dw_embed, inputs_qwm, inputs_dwm, raw_score=None):
        qwu_embed = torch.transpose(
            torch.squeeze(self.conv_uni(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1,
            2) + 0.000000001
        qwb_embed = torch.transpose(
            torch.squeeze(self.conv_bi(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1,
            2) + 0.000000001
        qwt_embed = torch.transpose(
            torch.squeeze(self.conv_tri(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1,
            2) + 0.000000001
        dwu_embed = torch.squeeze(self.conv_uni(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)),
                                  dim=3) + 0.000000001
        dwb_embed = torch.squeeze(self.conv_bi(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)),
                                  dim=3) + 0.000000001
        dwt_embed = torch.squeeze(self.conv_tri(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)),
                                  dim=3) + 0.000000001
        qwu_embed_norm = F.normalize(qwu_embed, p=2, dim=2, eps=1e-10)
        qwb_embed_norm = F.normalize(qwb_embed, p=2, dim=2, eps=1e-10)
        qwt_embed_norm = F.normalize(qwt_embed, p=2, dim=2, eps=1e-10)
        dwu_embed_norm = F.normalize(dwu_embed, p=2, dim=1, eps=1e-10)
        dwb_embed_norm = F.normalize(dwb_embed, p=2, dim=1, eps=1e-10)
        dwt_embed_norm = F.normalize(dwt_embed, p=2, dim=1, eps=1e-10)
        mask_qw = inputs_qwm.view(qw_embed.size()[0], qw_embed.size()[1], 1)
        mask_dw = inputs_dwm.view(dw_embed.size()[0], 1, dw_embed.size()[1], 1)
        mask_qwu = mask_qw[:, :qw_embed.size()[1] - (1 - 1), :]
        mask_qwb = mask_qw[:, :qw_embed.size()[1] - (2 - 1), :]
        mask_qwt = mask_qw[:, :qw_embed.size()[1] - (3 - 1), :]
        mask_dwu = mask_dw[:, :, :dw_embed.size()[1] - (1 - 1), :]
        mask_dwb = mask_dw[:, :, :dw_embed.size()[1] - (2 - 1), :]
        mask_dwt = mask_dw[:, :, :dw_embed.size()[1] - (3 - 1), :]
        log_pooling_sum_wwuu = self.get_intersect_matrix(qwu_embed_norm, dwu_embed_norm, mask_qwu, mask_dwu)
        log_pooling_sum_wwut = self.get_intersect_matrix(qwu_embed_norm, dwt_embed_norm, mask_qwu, mask_dwt)
        log_pooling_sum_wwub = self.get_intersect_matrix(qwu_embed_norm, dwb_embed_norm, mask_qwu, mask_dwb)
        log_pooling_sum_wwbu = self.get_intersect_matrix(qwb_embed_norm, dwu_embed_norm, mask_qwb, mask_dwu)
        log_pooling_sum_wwtu = self.get_intersect_matrix(qwt_embed_norm, dwu_embed_norm, mask_qwt, mask_dwu)
        log_pooling_sum_wwbb = self.get_intersect_matrix(qwb_embed_norm, dwb_embed_norm, mask_qwb, mask_dwb)
        log_pooling_sum_wwbt = self.get_intersect_matrix(qwb_embed_norm, dwt_embed_norm, mask_qwb, mask_dwt)
        log_pooling_sum_wwtb = self.get_intersect_matrix(qwt_embed_norm, dwb_embed_norm, mask_qwt, mask_dwb)
        log_pooling_sum_wwtt = self.get_intersect_matrix(qwt_embed_norm, dwt_embed_norm, mask_qwt, mask_dwt)

        log_pooling_sum = torch.cat(
            [log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu,
             log_pooling_sum_wwtu, \
             log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        if raw_score is None:
            output = torch.squeeze(self.dense_f(log_pooling_sum), 1)
        else:
            feature_pp = torch.cat([log_pooling_sum, raw_score.unsqueeze(1)], 1)
            output = torch.squeeze(self.dense_fpp(feature_pp), 1)
        return output, log_pooling_sum


def raw2tok(s):
    lst = regex_multi_space.sub(' ', regex_drop_char.sub(' ', s.lower())).strip().split()
    return lst

def load_glove(glove_file):
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

    with open(glove_file, 'r') as f:
        for line in f:
            val = line.split()
            idx2word.append(val[0])
            word2idx[val[0]] = idx
            word2vec[val[0]] = np.asarray(val[1:], dtype='float32')
            idx += 1

    return idx2word, word2idx, word2vec

def create_embeddings(idx2word, word2vec):
    embedding_matrix = np.zeros((len(idx2word), 300))
    for idx, word in enumerate(idx2word):
        embedding_matrix[idx] = word2vec[word]

    return embedding_matrix

class devFeatures(object):
    def __init__(self, query_id, doc_id, qd_score, query, doc, raw_score, query_idx, doc_idx, query_len, doc_len):
        self.query_id = query_id
        self.doc_id = doc_id
        self.qd_score = qd_score
        self.query = query
        self.doc = doc
        self.raw_score = raw_score
        self.query_idx = query_idx
        self.doc_idx = doc_idx
        self.query_len = query_len
        self.doc_len = doc_len

def tok2idx(toks, word2idx):
    input_ids = []
    for tok in toks:
        if tok in word2idx:
            input_ids.append(word2idx[tok])
        else:
            input_ids.append(word2idx['<UNK>'])
    return input_ids

def filter_sw(toks, length):
    wordsFiltered = []
    for w in toks:
        if w not in sws:
            w = stemmer.stem(w)
            if len(wordsFiltered) >= length:
                break
            wordsFiltered.append(w)
    return wordsFiltered

def read_data_to_features(input_file, word2idx, args):
    with open(input_file, 'r') as reader:
        cnt = 0
        features = []
        for line in reader:
            cnt += 1
            s = line.strip('\n').split('\t')

            qd_score = 0
            query_id = s[3]
            doc_id = s[4]
            raw_score = float(s[5])
            query_toks = filter_sw(raw2tok(s[0]), args.max_query_len)
            doc_toks = filter_sw(raw2tok(s[1]), args.max_doc_len)

            query_len = len(query_toks)
            doc_len = len(doc_toks)

            while len(query_toks) < 3:
                query_toks.append('<PAD>')
            while len(doc_toks) < 3:
                doc_toks.append('<PAD>')

            query_idx = tok2idx(query_toks, word2idx)
            doc_idx = tok2idx(doc_toks, word2idx)

            features.append(devFeatures(
                query_id = query_id,
                doc_id = doc_id,
                qd_score = qd_score,
                query = s[0],
                doc = s[1],
                raw_score = raw_score,
                query_idx = query_idx,
                doc_idx = doc_idx,
                query_len = query_len,
                doc_len = doc_len))

        return features

class Multi_Trans(nn.Module):
    def __init__(self, embedding_matrix, args):
        super(Multi_Trans, self).__init__()
        em = torch.tensor(embedding_matrix, dtype=torch.float32).cuda()
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.embedding.weight = nn.Parameter(em)
        self.embedding.weight.requires_grad = True
        self.ranker = cknrm(args.kernel_size, args.embedding_dim, args.cnn_kernel, args.cuda, True)
    
    def forward(self, query_idx, pos_idx, neg_idx, query_len, pos_len, neg_len, raw_score=None, is_training=True):
        query_embed = self.embedding(query_idx)
        doc_embed = self.embedding(pos_idx)
        query_mask = self.create_mask_like(query_len, query_embed)
        doc_mask = self.create_mask_like(pos_len, doc_embed)

        doc_scores, doc_features = self.ranker(query_embed, doc_embed, query_mask, doc_mask)#, raw_score)

        return doc_scores, doc_features
    
    def create_mask_like(self, lengths, like):
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask

def devDataLoader(features, batch_size):
    batches = []
    n_samples = len(features)
    idx = np.arange(n_samples)
    for start_idx in range(0, n_samples, batch_size):
        batch_idx = idx[start_idx:start_idx+batch_size]

        query_id = [features[i].query_id for i in batch_idx]
        doc_id = [features[i].doc_id for i in batch_idx]
        qd_score = [features[i].qd_score for i in batch_idx]
        query = [features[i].query for i in batch_idx]
        doc = [features[i].doc for i in batch_idx]
        raw_score = torch.tensor([features[i].raw_score for i in batch_idx], dtype=torch.float)
        query_idx = [torch.tensor(features[i].query_idx, dtype=torch.long) for i in batch_idx]
        doc_idx = [torch.tensor(features[i].doc_idx, dtype=torch.long) for i in batch_idx]
        query_len = torch.tensor([features[i].query_len for i in batch_idx], dtype=torch.long)
        doc_len = torch.tensor([features[i].doc_len for i in batch_idx], dtype=torch.long)

        query_idx = nn.utils.rnn.pad_sequence(query_idx, batch_first=True)
        doc_idx = nn.utils.rnn.pad_sequence(doc_idx, batch_first=True)

        batch = (query_id, doc_id, qd_score, query, doc, raw_score, query_idx, doc_idx, query_len, doc_len)
        batches.append(batch)
    return batches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', help='embedding path (glove embedding)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--test_file', help='test file path')
    parser.add_argument('--out_path', help='out path to save trec file')
    parser.add_argument('--pretrained_model', help='check point to load')
    parser.add_argument('--vocab_size', default=400002, type=int, help='vocab size with padding and unk words')
    parser.add_argument('--embedding_dim', default=300, type=int, help='embedding dim')
    parser.add_argument('--kernel_size', default=21, type=int, help='kernel size')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--cnn_kernel', default=128, type=int, help='cnn kernel size')
    parser.add_argument('--max_query_len', default=20, type=int, help='max query length')
    parser.add_argument('--max_doc_len', default=256, type=int, help='max doc length')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    idx2word, word2idx, word2vec = load_glove(args.embedding_path)
    embedding_matrix = create_embeddings(idx2word, word2vec)

    model = Multi_Trans(embedding_matrix, args)
    state_dict=torch.load(args.pretrained_model)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    model.to(device)

    # test data
    test_features = read_data_to_features(args.test_file, word2idx, args)
    test_data = devDataLoader(test_features, args.batch_size)

    # test
    rst_dict = {}
    for s, batch in enumerate(test_data):
        query_id = batch[0]
        doc_id = batch[1]
        qd_score = batch[2]
        query = batch[3]
        doc = batch[4]
        batch = tuple(t.to(device) for t in batch[5:])
        (raw_score, query_idx, doc_idx, query_len, doc_len) = batch

        with torch.no_grad():
            doc_scores, doc_features = model(query_idx, doc_idx, None, query_len, doc_len, None, raw_score, False)
        d_scores = doc_scores.detach().cpu().tolist()
        d_features = doc_features.detach().cpu().tolist()
        raw_score = raw_score.detach().cpu().tolist()
        for (q_id, d_id, qd_s, q, d, r_s, d_s, d_f) in zip(query_id, doc_id, qd_score, query, doc, raw_score, d_scores, d_features):
            if q_id in rst_dict:
                rst_dict[q_id].append((qd_s, d_s, d_id, q, d))
            else:
                rst_dict[q_id] = [(qd_s, d_s, d_id, q, d)]

    with open(args.out_path, 'w') as writer:
        for q_id, scores in rst_dict.items():
            ps = {}
            res = sorted(scores, key=lambda x: x[1], reverse=True)
            for rank, value in enumerate(res):
                if value[2] not in ps:
                    ps[value[2]] = 1
                    writer.write(q_id+' '+'Q0'+' '+str(value[2])+' '+str(len(ps))+' '+str(value[1])+' '+'Conv-KNRM'+'\n')



if __name__ == "__main__":
    main()
