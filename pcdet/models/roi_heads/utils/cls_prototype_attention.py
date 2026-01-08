import torch
from torch import nn as nn
from torch.nn import functional as F
import copy


def block_conv1d(in_channel, out_channel, kernel_size, padding, bias, inplace ):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=bias),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(inplace=inplace)
    )

class CLS_Attention_Layer(nn.Module):

    def __init__(self, dim_model, nhead, dropout, two_linear=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, num_heads=nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.norm_last = nn.BatchNorm1d(dim_model)

        self.linear_1 = block_conv1d(dim_model, dim_model, kernel_size=1,
            padding=0, bias=True, inplace=True)
        
        self.two_linear = two_linear
        if self.two_linear:
            self.linear_2 = block_conv1d(dim_model, dim_model, kernel_size=1,
                padding=0, bias=True, inplace=True)
        self.init_weights()


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def init_weights(self):
        for m in self.parameters():
            # print(m)
            if m.dim() > 1:
                nn.init.xavier_uniform_(m, gain=1)
        nn.init.constant_(self.norm_last.weight, 1)
        nn.init.constant_(self.norm_last.bias, 0)
    
    # def init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
    #         for m in module_list.modules():
    #             if isinstance(m, nn.Linear):
    #                 init_func(m.weight)
    #                 if m.bias is not None:
    #                     nn.init.constant_(m.bias, 0)
                    
    #     nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
    #     nn.init.constant_(self.cls_pred_layer.bias, 0)
    #     nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
    #     nn.init.constant_(self.reg_pred_layer.bias, 0)

    def feature_norm(self, feature):
        features_norm = torch.norm(feature, p=2, dim=1)
        feature = feature.div(features_norm.unsqueeze(1))
        return feature

    def forward(self, query_feat, prototypes, no_skip=False):
        query = query_feat.permute(1, 0, 2) # N,B,C
        key = value = prototypes.permute(1, 0, 2)
        query_reweighted = self.self_attn(query, key, value=value)[0]
        if not no_skip:
            query = query + self.dropout1(query_reweighted)
        else:
            query = self.dropout1(query_reweighted)
        query = query.permute(1, 2, 0) # B,C,N
        query = self.norm_last(query)
        if self.two_linear:
            query = self.linear_2(self.linear_1(query))
        else:
            query = self.linear_1(query)
        query = query.permute(0, 2, 1) # B,N,C
        return query


class CLS_Attention_Layer_VAE(nn.Module):

    def __init__(self, dim_model, nhead, dropout, num_cls, mu, two_linear=False):
        super().__init__()
        self.num_cls = num_cls
        self.dim_model = dim_model
        self.mu = mu

        self.memory_banks_embedding = torch.nn.Embedding(num_embeddings=self.num_cls, embedding_dim=self.dim_model).cuda()
        self.memory_banks = self.memory_banks_embedding.weight
        self.memory_banks_embedding.weight.requires_grad = False
        self.memory_banks_embedding.requires_grad = False

        self.self_attn = nn.MultiheadAttention(dim_model, num_heads=nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.norm_last = nn.BatchNorm1d(dim_model)

        self.linear_1 = block_conv1d(dim_model, dim_model, kernel_size=1,
            padding=0, bias=True, inplace=True)
        
        self.two_linear = two_linear
        if self.two_linear:
            self.linear_2 = block_conv1d(dim_model, dim_model, kernel_size=1,
                padding=0, bias=True, inplace=True)
        self.init_weights()


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def init_weights(self):
        for m in self.parameters():
            # print(m)
            if m.dim() > 1:
                nn.init.xavier_uniform_(m, gain=1)
        nn.init.constant_(self.norm_last.weight, 1)
        nn.init.constant_(self.norm_last.bias, 0)
        self.memory_banks_embedding.weight[:] = torch.rand(self.num_cls, self.dim_model)
    

    # def feature_norm(self, feature):
    #     features_norm = torch.norm(feature, p=2, dim=1)
    #     feature = feature.div(features_norm.unsqueeze(1))
    #     return feature

    def update_memory(self, info):
        '''
        args:   
            info:
                labels_feat: 
                feats_for_update:
        '''
        
        feature_list = [[] for i in range(self.memory_banks.shape[0])]

        for index in range(len(feature_list)):
            flag = info['labels_feat'] == (index+1)
            num_fg_feats = flag.sum()
            if num_fg_feats > 0:
                feature_list[index].append(info['feats_for_update'][flag].view(num_fg_feats, -1).clone().detach())

        for index in range(len(feature_list)):
            # print(index, len(feature_list[index]))
            if len(feature_list[index]) == 0:
                continue

            feature_list[index] = torch.cat(feature_list[index], dim=0)
            one_slot = torch.mean(feature_list[index], dim=0)
            features_norm = torch.norm(one_slot, p=2, dim=0)
            one_slot = one_slot.div(features_norm)
            self.memory_banks[index] = self.mu * self.memory_banks[index] + (1 - self.mu) * one_slot

        features_norm = torch.norm(self.memory_banks_embedding.weight, p=2, dim=1).unsqueeze(1)
        self.memory_banks_embedding.weight[:] = self.memory_banks_embedding.weight.div(features_norm)
        return feature_list

    def forward(self, query_feat, prototype_info=None):

        if prototype_info is not None:
            features_for_update = self.update_memory(prototype_info)

        query = query_feat.permute(1, 0, 2) # N,B,C
        key = value = self.memory_banks.unsqueeze(1).clone()

        query_reweighted = self.self_attn(query, key, value=value)[0]
        
        query = query + self.dropout1(query_reweighted)
        query = query.permute(1, 2, 0) # B,C,N
        query = self.norm_last(query)
        if self.two_linear:
            query = self.linear_2(self.linear_1(query))
        else:
            query = self.linear_1(query)
        query = query.permute(0, 2, 1) # B,N,C

        if prototype_info is not None:
            return query, features_for_update
        else:
            return query
