"""
ApexOracle Model Layers

This module contains neural network layers used in antibiotic property prediction.
Extracted from antibiotic_classifier.py for cleaner dependency management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RegressionHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        hidden_dim_1 = 384,
        hidden_dim_2 = 128,
        num_targets = 19,
        pooler_dropout: float=0.2,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x



class FirstTokenAttention_genome(nn.Module):
    def __init__(self, mol_cls_embed_dim, genome_embed_dim, num_heads, dropout=0.1):
        super(FirstTokenAttention_genome, self).__init__()
        self.mol_to_genome_dim = nn.Linear(mol_cls_embed_dim, genome_embed_dim)
        # self.genome_to_mol_dim = nn.Linear(genome_embed_dim, mol_cls_embed_dim)
        # 多头注意力层
        self.key_value_projection = nn.Linear(genome_embed_dim, genome_embed_dim * 2)
        self.mha = nn.MultiheadAttention(genome_embed_dim, num_heads, dropout=dropout)
        # 残差和归一化（LayerNorm）
        self.attn_norm = nn.LayerNorm(genome_embed_dim)
        self.norm1 = nn.LayerNorm(genome_embed_dim)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(genome_embed_dim, genome_embed_dim),
            nn.GELU(),
            nn.Linear(genome_embed_dim, genome_embed_dim)
        )
        self.norm2 = nn.LayerNorm(genome_embed_dim)

    def forward(self, mol_cls_emb, genome_embs, key_padding_mask, **kwargs):
        """
        x: Tensor, shape = (batch_size, seq_len, embed_dim)
        """
        # 提取序列的第一个 token，作为 query，形状: (batch_size, 1, embed_dim)
        genome_embs_dim = genome_embs.shape[-1]
        query = self.mol_to_genome_dim(mol_cls_emb)[:, None, :]

        if torch.isnan(query).any():
            print(" query 中包含 NaN\n")

        # nn.MultiheadAttention 要求输入 shape 为 (seq_len, batch_size, embed_dim)
        query = query.transpose(0, 1)  # (1, batch_size, embed_dim)
        key_value = self.key_value_projection(genome_embs.reshape(-1, genome_embs.shape[-1])).reshape([genome_embs.shape[0], genome_embs.shape[1], -1])
        key_value = key_value.transpose(0, 1)  # (seq_len, batch_size, embed_dim)

        if torch.isnan(key_value).any():
            print(" key_value 中包含 NaN\n")

        # value = key
        query_norm = self.attn_norm(query.squeeze(0)).unsqueeze(0)
        # 计算多头注意力：只计算第一个 token 对整个序列的注意力
        attn_output, attn_weights = self.mha(query_norm, key_value[:, :, :genome_embs_dim], key_value[:, :, genome_embs_dim:], key_padding_mask = key_padding_mask.to(torch.bool))  # (1, batch_size, embed_dim)

        if torch.isnan(attn_output).any():
            print(" attn_output 中包含 NaN\n")
            print(key_padding_mask)
            print(key_padding_mask.shape)
            print(f' sum: {key_padding_mask.sum()}')
            exit(0)

        # 残差连接与归一化
        # attn_output = self.genome_to_mol_dim(attn_output.squeeze())
        query = self.norm1(query.squeeze() + attn_output.squeeze())

        # 前馈网络 + 残差连接和归一化
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)

        # 最终只输出更新后的第一个 token embedding，返回形状 (batch_size, embed_dim)
        return query





# 总行数: 110
