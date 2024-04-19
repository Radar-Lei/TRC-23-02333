import numpy as np
import torch
import torch.nn as nn
from submodules import VariEmbedding, Conv1d_with_init, TranResidualBlock
import torch.fft
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, configs, inputdim = 2):
        super(Model, self).__init__()
        self.configs = configs
        
        self.cond_embedding = VariEmbedding(configs)
        
        self.trans_inp_proj = Conv1d_with_init(inputdim, configs.trans_channels, 1)
        
        self.trans_layers = nn.ModuleList(
            [
                TranResidualBlock(
                    side_dim=1+configs.spa_pos_emb_dim+configs.d_model,
                    channels=configs.trans_channels,
                    fusion_d = configs.d_model + configs.trans_channels + configs.c_out, # channels+tn_d_model+K                    
                    nheads=configs.nheads,
                    configs = configs
                )
                for _ in range(configs.trans_layers)
            ]
        )
        
        self.output_projection1 = Conv1d_with_init(configs.trans_channels, configs.trans_channels, 1)
        self.output_projection2 = Conv1d_with_init(configs.trans_channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)


    def forward(self, inp, timestamps, mask=None, target_mask=None, weight_A=None):
        """
        inp: traffic state data (unembedded) of shape (B, L, K)
        timestamps: timestamps (unembedded) of shape (B, L, ?)
        mask: randomly created mask of shape (B, L, K)
        target_mask: fixed mask, concerning original missing and fixed missing for test, of shape (B, L, K)
        """
        B,L,K = inp.shape
        
        # inp_emb is of shape (B, d_model, K, L)
        # tem_emb is of shape (B, L, d_model), timestamps embedding
        # tem_pos_emb is of shape (1, L, d_model), timestamps position embedding
        # spa_pos_emb is of shape (B, spa_pos_emb_dim, K, L)
        inp_emb, tem_emb, tem_pos_emb, spa_pos_emb  = self.cond_embedding(inp, timestamps)
        
        # embedding and concat side info for transformer module
        ext_mask = mask.permute(0,2,1).unsqueeze(1) # (B,1,K,L)
        
        # (B, K, K) -> (B, K, K, 1) -> (B, K, K, L)
        extra_spa_fea = weight_A.unsqueeze(-1).expand(-1, -1, -1, L)
        
        #side_info is of shape (B, 1+spa_pos_emb_dim+d_model, K, L)
        side_info = torch.cat([ext_mask, spa_pos_emb, tem_pos_emb], dim=1)
        
        skip = []
        for layer in self.trans_layers:
            inp_emb, skip_connection = layer(
                inp_emb,
                side_info,
                tem_emb,
                extra_spa_fea,
                )
            skip.append(skip_connection)
            
        pred = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.trans_layers))
        pred = pred.reshape(B, self.configs.trans_channels, K * L)
        pred = self.output_projection1(pred)  # (B,channel,K*L)
        pred = F.relu(pred)
        pred = self.output_projection2(pred)  # (B,1,K*L)
        pred = pred.reshape(B, K, L).permute(0, 2, 1) # (B, 1, K*L) -> (B, K, L) -> (B,L,K)

        return pred, inp