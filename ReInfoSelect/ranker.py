import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *
from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder,MultiHeadSelfAttention
from transformers import *

from utils import *

class knrm(nn.Module):
    def __init__(self, cfg, embedding_init=None):
        super(knrm, self).__init__()
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

        feature_dim = cfg["n_kernels"]
        if cfg["score_feature"]:
            feature_dim = cfg["n_kernels"] + 1
        self.dense = nn.Linear(feature_dim, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q.type_as(self.sigma) * 0.01
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def forward(self, q_embed, d_embed, mask_q, mask_d, score_feature=None):
        q_embed_norm = F.normalize(q_embed, 2, 2)
        d_embed_norm = F.normalize(d_embed, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)

        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        if score_feature is not None:
            log_pooling_sum = torch.cat([log_pooling_sum, score_feature.unsqueeze(1)], 1)
        score = torch.squeeze(self.dense(log_pooling_sum), 1)
        return score, log_pooling_sum

class cknrm(nn.Module):
    def __init__(self, cfg, embedding_init=None):
        super(cknrm, self).__init__()
        self.d_word_vec = cfg["embedding_dim"]
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

        self.tanh = nn.Tanh()
        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, cfg["embedding_dim"])),
            nn.ReLU()
        )
        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, cfg["embedding_dim"])),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, cfg["embedding_dim"])),
            nn.ReLU()
        )

        feature_dim = cfg["n_kernels"] * 9
        if cfg["score_feature"]:
            feature_dim = cfg["n_kernel"] * 9 + 1
        self.dense = nn.Linear(feature_dim, 1)

    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * atten_q.type_as(self.sigma)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def forward(self, qw_embed, dw_embed, inputs_qwm, inputs_dwm, score_feature=None):
        qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        dwu_embed = torch.squeeze(self.conv_uni(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        dwb_embed = torch.squeeze(self.conv_bi (dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        dwt_embed = torch.squeeze(self.conv_tri(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
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

        log_pooling_sum = torch.cat([ log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub, log_pooling_sum_wwbu, log_pooling_sum_wwtu,\
            log_pooling_sum_wwbb, log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt], 1)
        if score_feature is not None:
            log_pooling_sum = torch.cat([log_pooling_sum, score_feature.unsqueeze(1)], 1)
        score = self.dense(log_pooling_sum).squeeze(-1)
        return score, log_pooling_sum

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

class edrm(nn.Module):
    def __init__(self, cfg, embedding_init=None):
        super(edrm, self).__init__()
        self.d_word_vec = cfg["embedding_dim"]
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

        self.entemb = nn.Embedding(cfg["ent_size"], 128)
        if entemb is not None:
            em = torch.tensor(entemb_init, dtype=torch.float32)
            self.entemb.weight = nn.Parameter(em)
            self.entemb.weight.requires_grad = True

        self.ent_q_size= 5
        self.ent_d_size = 10

        self.tanh = nn.Tanh()
        self.conv_des = nn.Sequential(
            nn.Conv1d(1, 128, cfg["embedding_dim"] * 5, stride=cfg["embedding_dim"]),
            nn.ReLU(),
            nn.MaxPool1d(20 - 5 + 1),
        )
        self.conv_uni = nn.Sequential(
            nn.Conv2d(1, 128, (1, cfg["embedding_dim"])),
            nn.ReLU()
        )
        self.conv_bi = nn.Sequential(
            nn.Conv2d(1, 128, (2, cfg["embedding_dim"])),
            nn.ReLU()
        )
        self.conv_tri = nn.Sequential(
            nn.Conv2d(1, 128, (3, cfg["embedding_dim"])),
            nn.ReLU()
        )

        feature_dim = cfg["n_kernels"] * 16
        if cfg["score_feature"]:
            feature_dim = cfg["n_kernel"] * 16 + 1
        self.dense = nn.Linear(feature_dim, 1)

    def get_intersect_matrix(self, q_embed, d_embed, atten_q, atten_d):
        sim = torch.bmm(q_embed, d_embed).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[2], 1)
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * atten_d.type_as(self.sigma)
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * atten_q.type_as(self.sigma)
        log_pooling_sum = torch.sum(log_pooling_sum, 1)
        return log_pooling_sum

    def forward(self, qw_embed, dw_embed, inputs_qwm, inputs_dwm, qi_embed, qe_embed, di_embed, de_embed, inputs_qem, inputs_dem, raw_score=None):
        self.batch_size = qw_embed.size()[0]

        qe_embed_conv = self.conv_des(qe_embed.view(self.batch_size * self.ent_q_size, 1, -1))
        de_embed_conv = self.conv_des(de_embed.view(self.batch_size * self.ent_d_size, 1, -1))
        qe = qe_embed_conv.view(self.batch_size, -1, 128)
        de = de_embed_conv.view(self.batch_size, -1, 128)
        qi = qi_embed.view(self.batch_size, -1, 128)
        di = di_embed.view(self.batch_size, -1, 128)
        qs_embed = qi + qe
        ds_embed = di + de
        qeu_embed_norm = F.normalize(qs_embed, p=2, dim=2, eps=1e-10)
        deu_embed_norm = torch.transpose(F.normalize(ds_embed, p=2, dim=2, eps=1e-10), 1, 2)
        mask_qeu = inputs_qem.view(qi.size()[0], qi.size()[1], 1)
        mask_deu = inputs_dem.view(di.size()[0], 1, di.size()[1], 1)

        qwu_embed = torch.transpose(torch.squeeze(self.conv_uni(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        qwb_embed = torch.transpose(torch.squeeze(self.conv_bi (qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        qwt_embed = torch.transpose(torch.squeeze(self.conv_tri(qw_embed.view(qw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3), 1, 2) + 0.000000001
        dwu_embed = torch.squeeze(self.conv_uni(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        dwb_embed = torch.squeeze(self.conv_bi (dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
        dwt_embed = torch.squeeze(self.conv_tri(dw_embed.view(dw_embed.size()[0], 1, -1, self.d_word_vec)), dim=3) + 0.000000001
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
        log_pooling_sum_ewuu = self.get_intersect_matrix(qeu_embed_norm, dwu_embed_norm, mask_qeu, mask_dwu)
        log_pooling_sum_ewub = self.get_intersect_matrix(qeu_embed_norm, dwb_embed_norm, mask_qeu, mask_dwb)
        log_pooling_sum_ewut = self.get_intersect_matrix(qeu_embed_norm, dwt_embed_norm, mask_qeu, mask_dwt)
        log_pooling_sum_weuu = self.get_intersect_matrix(qwu_embed_norm, deu_embed_norm, mask_qwu, mask_deu)
        log_pooling_sum_webu = self.get_intersect_matrix(qwb_embed_norm, deu_embed_norm, mask_qwb, mask_deu)
        log_pooling_sum_wetu = self.get_intersect_matrix(qwt_embed_norm, deu_embed_norm, mask_qwt, mask_deu)
        log_pooling_sum_eeuu = self.get_intersect_matrix(qeu_embed_norm, deu_embed_norm, mask_qeu, mask_deu)

        log_pooling_sum = torch.cat([log_pooling_sum_wwuu, log_pooling_sum_wwut, log_pooling_sum_wwub,
                                     log_pooling_sum_wwbu, log_pooling_sum_wwtu, log_pooling_sum_wwbb,
                                     log_pooling_sum_wwbt, log_pooling_sum_wwtb, log_pooling_sum_wwtt,
                                     log_pooling_sum_ewuu, log_pooling_sum_ewub, log_pooling_sum_ewut,
                                     log_pooling_sum_weuu, log_pooling_sum_webu, log_pooling_sum_wetu, log_pooling_sum_eeuu
                                     ], 1)
        if score_feature is not None:
            log_pooling_sum = torch.cat([log_pooling_sum, score_feature.unsqueeze(1)], 1)
        score = self.dense(log_pooling_sum).squeeze(-1)
        return score, log_pooling_sum

class BertForRanking(nn.Module):
    def __init__(self, cfg):
        super(BertForRanking, self).__init__()
        config = BertConfig.from_pretrained(cfg["config"])
        self.bert = BertModel.from_pretrained(cfg["model"], config=config)
        self.bert.train()

        feature_dim = config.hidden_size
        if cfg["score_feature"]:
            feature_dim = config.hidden_size + 1
        self.dense = nn.Linear(feature_dim, 1)

    def forward(self, inst, tok, mask, score_feature=None):
        output = self.bert(inst, token_type_ids=tok, attention_mask=mask)
        if score_feature is not None:
            logits = torch.cat([output[1], score_feature.unsqueeze(1)], 1)
        score = torch.squeeze(self.dense(output[1]), dim=1)
        return score, output[1]
