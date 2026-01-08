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

class FEAT_Attention_Layer(nn.Module):

    def __init__(self, dim_model, nhead, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim_model, num_heads=nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout, inplace=True)

        self.norm_last = nn.BatchNorm1d(dim_model)

        self.linear_1 = block_conv1d(dim_model, dim_model, kernel_size=1,
            padding=0, bias=True, inplace=True)

        # self.linear_2 = block_conv1d(dim_model, dim_model, kernel_size=1,
        #     padding=0, bias=True, inplace=True)
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

    def forward(self, query_feat, prototypes, pos_query=None, pos_key=None):
        query = query_feat.permute(1, 0, 2) if pos_query is None \
            else (query_feat + pos_query).permute(1, 0, 2) # N,B,C

        key = prototypes.permute(1, 0, 2) if pos_key is None \
            else (prototypes + pos_key).permute(1, 0, 2)
        value = prototypes.permute(1, 0, 2)
        query_reweighted = self.self_attn(query, key, value=value)[0]
        query = query + self.dropout1(query_reweighted)
        query = query.permute(1, 2, 0) # B,C,N
        query = self.norm_last(query)
        # query = self.linear_2(self.linear_1(query))
        query = self.linear_1(query)
        query = query.permute(0, 2, 1) # B,N,C
        return query
