import torch
import torch.nn as nn
from torch.autograd import Variable

from model_utils import kernal_mus, kernel_sigmas

from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *                          
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder,MultiHeadSelfAttention

class tk(nn.Module):
    def __init__(self, cfg, embedding_init=None):
        super(tk, self).__init__()
        tensor_mu = torch.FloatTensor(kernal_mus(cfg["n_kernels"]))
        tensor_sigma = torch.FloatTensor(kernel_sigmas(cfg["n_kernels"]))
        if torch.cuda.is_available():
            tensor_mu = tensor_mu.cuda()
            tensor_sigma = tensor_sigma.cuda()
        self.mu = Variable(tensor_mu, requires_grad=False).view(1, 1, 1, cfg["n_kernels"])
        self.sigma = Variable(tensor_sigma, requires_grad=False).view(1, 1, 1, cfg["n_kernels"])

        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        if embedding_init is not None:
            em = torch.tensor(embedding_init, dtype=torch.float32)
            self.embedding.weight = nn.Parameter(em)
            self.embedding.weight.requires_grad = True

        self.nn_scaler = nn.Parameter(torch.full([1], 0.01, dtype=torch.float32, requires_grad=True))
        self.mixer = nn.Parameter(torch.full([1,1,1], 0.5, dtype=torch.float32, requires_grad=True))
        self.stacked_att = StackedSelfAttentionEncoder(input_dim=cfg["embedding_dim"],
                 hidden_dim=cfg["embedding_dim"],
                 projection_dim=32,
                 feedforward_hidden_dim=100,
                 num_layers=2,
                 num_attention_heads=8,
                 dropout_prob = 0,
                 residual_dropout_prob = 0,
                 attention_dropout_prob = 0)
        self.cosine_module = CosineMatrixAttention()
        self.dense = nn.Linear(cfg["n_kernels"], 1, bias=False)
        #self.dense_mean = nn.Linear(num_kernels, 1, bias=False)
        #self.dense_comb = nn.Linear(2, 1, bias=False)
        self.dense_fpp = nn.Linear(cfg["n_kernels"] + 1, 1, bias=False)# toby add

        torch.nn.init.uniform_(self.dense.weight, -0.014, 0.014)  # inits taken from matchzoo
        #torch.nn.init.uniform_(self.dense_mean.weight, -0.014, 0.014)  # inits taken from matchzoo
        torch.nn.init.uniform_(self.dense_fpp.weight, -0.014, 0.014)  # inits taken from matchzoo

    def forward(self, q_embed, d_embed, mask_q, mask_d, raw_score=None):
        q_embed = q_embed * mask_q.unsqueeze(-1)
        d_embed = d_embed * mask_d.unsqueeze(-1)

        q_embed_context = self.stacked_att(q_embed, mask_q)
        d_embed_context = self.stacked_att(d_embed, mask_d)

        q_embed = (self.mixer * q_embed + (1 - self.mixer) * q_embed_context) * mask_q.unsqueeze(-1)
        d_embed = (self.mixer * d_embed + (1 - self.mixer) * d_embed_context) * mask_d.unsqueeze(-1)

        q_by_d_mask = torch.bmm(mask_q.unsqueeze(-1), mask_d.unsqueeze(-1).transpose(-1, -2))
        q_by_d_mask_view = q_by_d_mask.unsqueeze(-1)

        cosine_matrix = self.cosine_module.forward(q_embed, d_embed)
        cosine_matrix_masked = cosine_matrix * q_by_d_mask
        cosine_matrix_extradim = cosine_matrix_masked.unsqueeze(-1)

        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix_extradim - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * q_by_d_mask_view

        doc_lengths = torch.sum(mask_d, 1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        log_per_kernel_query_masked = log_per_kernel_query * mask_q.unsqueeze(-1) # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1)

        per_kernel_query_mean = per_kernel_query / (doc_lengths.view(-1,1,1) + 1) # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = log_per_kernel_query_mean * mask_q.unsqueeze(-1) # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1)

        dense_out = self.dense(per_kernel)
        #dense_mean_out = self.dense_mean(per_kernel_mean)
        #dense_comb_out = self.dense_comb(torch.cat([dense_out,dense_mean_out],dim=1))
        #score = torch.squeeze(dense_comb_out,1) #torch.tanh(dense_out), 1)

        if raw_score is None:
            score = torch.squeeze(dense_out,1)
        else:
            feature_pp = torch.cat([per_kernel, raw_score.unsqueeze(1)], 1)
            output = torch.squeeze(self.dense_fpp(feature_pp), 1)
        return score
