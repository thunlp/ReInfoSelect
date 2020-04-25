import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def kernal_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class cknrm(nn.Module):
    def __init__(self, args, embedding_init=None):
        super(cknrm, self).__init__()
        self.d_word_vec = args.embed_dim
        tensor_mu = torch.FloatTensor(kernal_mus(args.n_kernels))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(args.n_kernels))
        if torch.cuda.is_available():
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, args.n_kernels)
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, args.n_kernels)
        
        self.embedding = nn.Embedding(args.vocab_size, args.embed_dim)
        if embedding_init is not None:
            em = torch.tensor(embedding_init, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(em)
            self.embedding.weight.requires_grad = True

        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, args.embed_dim)),
            nn.ReLU()
        )
        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, args.embed_dim)),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, args.embed_dim)),
            nn.ReLU()
        )

        feature_dim = args.n_kernels * 9
        self.dense = nn.Linear(feature_dim, 1)
        self.dense_p = nn.Linear(feature_dim + 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * atten_q.type_as(self.sigma)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def create_mask_like(self, lengths, like):
        mask = torch.zeros(like.size()[:2])
        for ind, _length in enumerate(lengths.data):
            mask[ind, :_length] = 1
        mask = mask.type_as(like.data)
        mask = torch.autograd.Variable(mask, requires_grad=False)
        return mask

    def forward(self, query_idx, doc_idx, query_len, doc_len, raw_score=None):
        qw_embed = self.embedding(query_idx)
        dw_embed = self.embedding(doc_idx)
        inputs_qwm = self.create_mask_like(query_len, qw_embed)
        inputs_dwm = self.create_mask_like(doc_len, dw_embed)

        qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 1e-10
        qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 1e-10
        qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 1e-10
        dwu_embed = torch.squeeze(self.conv_uni(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 1e-10
        dwb_embed = torch.squeeze(self.conv_bi (dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 1e-10
        dwt_embed = torch.squeeze(self.conv_tri(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 1e-10
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
        
        log_pooling_sum = torch.cat([log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu, log_pooling_sum_wwtu,\
            log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        if raw_score is not None:
            log_pooling_sum = torch.cat([log_pooling_sum, raw_score.unsqueeze(1)], 1)
            score = self.dense_p(log_pooling_sum).squeeze(-1)
        else:
            score = self.dense(log_pooling_sum).squeeze(-1)
        return score, log_pooling_sum
