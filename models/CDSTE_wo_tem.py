import numpy as np
import torch
import torch.nn as nn
from submodules import SpaObsEmbedding, Conv1d_with_init, SpaResidualBlock
import torch.fft
import torch.nn.functional as F
import math

class Model(nn.Module):
    def __init__(self, configs, inputdim = 2):
        super(Model, self).__init__()
        self.configs = configs
        
        if configs.diff_schedule == "quad":
            self.beta = np.linspace(
                configs.beta_start ** 0.5, configs.beta_end ** 0.5, configs.diff_steps
            ) ** 2
        elif configs.diff_schedule == "linear":
            self.beta = np.linspace(
                configs.beta_start, configs.beta_end, configs.diff_steps
            )
        # self.beta is of shape (T,)
        self.alpha = 1 - self.beta
        """
        self.alpha_hat: cumulative product of alpha, e.g., alpha_hat[i] = alpha[0] * alpha[1] * ... * alpha[i]
        """
        self.alpha_hat = np.cumprod(self.alpha) # self.alpha is still of shape (T,)
        # reshape for computing, self.alpha_torch is of shape (T,) -> (T,1,1)
        self.alpha_torch = torch.tensor(self.alpha_hat).float().to(configs.gpu).unsqueeze(1).unsqueeze(1)

        self.cond_embedding = SpaObsEmbedding(configs.locations,
                                            configs.d_model,
                                            configs.embed, # timestamps embedding type
                                            configs.freq, # timestamps embedding frequency
                                            configs.dropout,
                                            configs.diff_steps,
                                            configs.diff_emb_dim
                                            )
        
        self.trans_inp_proj = Conv1d_with_init(inputdim, configs.channels, 1)
        # embedding for features, i.e., different locations
        self.spa_pos_emb = nn.Embedding(configs.locations, configs.spa_pos_emb_dim)
        
        self.trans_layers = nn.ModuleList(
            [
                SpaResidualBlock(
                    side_dim=1+configs.spa_pos_emb_dim+configs.d_model,
                    channels=configs.channels,
                    fusion_d = configs.d_model + configs.channels + configs.locations, # channels+tn_d_model+K                    
                    diffusion_embedding_dim=configs.diff_emb_dim, 
                    nheads=configs.nheads,
                    configs = configs
                )
                for _ in range(configs.layers)
            ]
        )
        
        self.output_projection1 = Conv1d_with_init(configs.channels, configs.channels, 1)
        self.output_projection2 = Conv1d_with_init(configs.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)


    def forward(self, x_enc, x_mark_enc, mask=None, target_mask=None, weight_A=None):
        """
        x_enc: traffic state data (unembedded) of shape (B, L, K)
        x_mark_enc: timestamps (unembedded) of shape (B, L, ?)
        mask: randomly created mask of shape (B, L, K)
        target_mask: fixed mask, concerning original missing and fixed missing for test, of shape (B, L, K)
        """
        B,_,_ = x_enc.shape

        t = torch.randint(0, self.configs.diff_steps, [B]).to(self.configs.gpu)

        # alpha_torch is of shape (T,1,1), t is of torch.Size([B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(x_enc) # (B,L,K)
        noisy_data = (current_alpha ** 0.5) * x_enc + ((1.0 - current_alpha) ** 0.5) * noise

        cond_obs = mask * x_enc
        mask = mask * target_mask
        noisy_target = (1-mask) * noisy_data
        
        # emb_cond_obs is of shape (B, L, d_model)
        # emb_tem is of shape (B, L, d_model), timestamps embedding
        # emb_tem_pos is of shape (1, L, d_model), timestamps position embedding
        # emb_diff_step is of shape (B, diff_emb_dim)
        emb_tem, emb_tem_pos, emb_diff_step  = self.cond_embedding(cond_obs, x_mark_enc, t)
        
        # the total input of transformer module is of shape (B,2,K,L)
        trans_total_inp = torch.cat([cond_obs.permute(0,2,1).unsqueeze(1), noisy_target.permute(0,2,1).unsqueeze(1)], dim=1)
        
        _, inputdim, K, L = trans_total_inp.shape
        
        trans_total_inp = trans_total_inp.reshape(B, inputdim, K * L)
        trans_total_inp = self.trans_inp_proj(trans_total_inp)
        trans_total_inp = F.relu(trans_total_inp)
        trans_total_inp = trans_total_inp.reshape(B, self.configs.channels, K, L)
        
        # embedding and concat side info for transformer module
        ext_mask = mask.permute(0,2,1).unsqueeze(1) # (B,1,K,L)
        spa_pos_emb = self.spa_pos_emb(
            torch.arange(K).to(self.configs.gpu)
            ) # (K,spa_pos_emb_dim)
        # # convert timestamp_emb from (B, L, d_model) to (B, L, K, d_model)
        # timestamp_emb = timestamp_emb.unsqueeze(2).expand(-1, -1, K, -1)
        
        # convert feature_emb from (K,spa_pos_emb_dim) to (B, L, K, spa_pos_emb_dim) -> (B, spa_pos_emb_dim, K, L)
        spa_pos_emb = spa_pos_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).permute(0,3,2,1)
        
        # convert emb_tem from (B, L, d_model) to (B, L, K, d_model) -> (B, d_model, K, L)
        emb_tem = emb_tem.unsqueeze(2).expand(-1, -1, K, -1).permute(0,3,2,1)
        
        # (B, K, K) -> (B, K, K, 1) -> (B, K, K, L)
        extra_spa_fea = weight_A.unsqueeze(-1).expand(-1, -1, -1, L)
        
        # emb_tem_pos is of shape (1, L, d_model) -> (1, 1, L, d_model) -> (B, K, L, d_model) -> (B, d_model, K, L)
        emb_tem_pos = emb_tem_pos.unsqueeze(0).expand(B, K, -1, -1).permute(0, 3, 1, 2)
        
        #side_info is of shape (B, 1+spa_pos_emb_dim+d_model, K, L)
        side_info = torch.cat([ext_mask, spa_pos_emb, emb_tem_pos], dim=1)
        
        skip = []
        for layer in self.trans_layers:
            trans_total_inp, skip_connection = layer(
                trans_total_inp,
                side_info,
                emb_tem,
                extra_spa_fea,
                emb_diff_step
                )
            skip.append(skip_connection)
            
        pred = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.trans_layers))
        pred = pred.reshape(B, self.configs.channels, K * L)
        pred = self.output_projection1(pred)  # (B,channel,K*L)
        pred = F.relu(pred)
        pred = self.output_projection2(pred)  # (B,1,K*L)
        pred = pred.reshape(B, K, L).permute(0, 2, 1) # (B, 1, K*L) -> (B, K, L) -> (B,L,K)

        return pred, noise
    

    def evaluate(self, x_enc, x_mark_enc, mask=None, target_mask=None, weight_A=None):
        B,L,K = x_enc.shape
        imputed_samples = torch.zeros(B, self.configs.diff_samples, L, K).to(self.configs.gpu)

        for i in range(self.configs.diff_samples):
            # diffusion starts from a pure Gaussian noise
            sample = torch.randn_like(x_enc)

            # initial diffusion step start from N-1
            for s in range(self.configs.diff_steps - 1, -1, -1):
                cond_obs = mask * x_enc
                noisy_target = target_mask * sample

                # emb_cond_obs is of shape (B, L, d_model)
                # emb_tem is of shape (B, L, d_model), timestamps embedding
                # emb_tem_pos is of shape (1, L, d_model), timestamps position embedding
                # emb_diff_step is of shape (B, diff_emb_dim)
                emb_cond_obs, emb_tem, emb_tem_pos, emb_diff_step  = self.cond_embedding(cond_obs, x_mark_enc, torch.tensor([s]).to(self.configs.gpu))

                # the total input of transformer module is of shape (B,2,K,L)
                trans_total_inp = torch.cat([cond_obs.permute(0,2,1).unsqueeze(1), noisy_target.permute(0,2,1).unsqueeze(1)], dim=1)
                
                _, inputdim, K, L = trans_total_inp.shape
                
                trans_total_inp = trans_total_inp.reshape(B, inputdim, K * L)
                trans_total_inp = self.trans_inp_proj(trans_total_inp)
                trans_total_inp = F.relu(trans_total_inp)
                trans_total_inp = trans_total_inp.reshape(B, self.configs.channels, K, L)
                
                # embedding and concat side info for transformer module
                ext_mask = mask.permute(0,2,1).unsqueeze(1) # (B,1,K,L)
                spa_pos_emb = self.spa_pos_emb(
                    torch.arange(K).to(self.configs.gpu)
                    ) # (K,spa_pos_emb_dim)
                # # convert timestamp_emb from (B, L, d_model) to (B, L, K, d_model)
                # timestamp_emb = timestamp_emb.unsqueeze(2).expand(-1, -1, K, -1)
                
                # convert feature_emb from (K,spa_pos_emb_dim) to (B, L, K, spa_pos_emb_dim) -> (B, spa_pos_emb_dim, K, L)
                spa_pos_emb = spa_pos_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).permute(0,3,2,1)
                
                # convert emb_tem from (B, L, d_model) to (B, L, K, d_model) -> (B, d_model, K, L)
                emb_tem = emb_tem.unsqueeze(2).expand(-1, -1, K, -1).permute(0,3,2,1)
                
                # (B, K, K) -> (B, K, K, 1) -> (B, K, K, L)
                extra_spa_fea = weight_A.unsqueeze(-1).expand(-1, -1, -1, L)
                
                # emb_tem_pos is of shape (1, L, d_model) -> (1, 1, L, d_model) -> (B, K, L, d_model) -> (B, d_model, K, L)
                emb_tem_pos = emb_tem_pos.unsqueeze(0).expand(B, K, -1, -1).permute(0, 3, 1, 2)
                
                #side_info is of shape (B, 1+spa_pos_emb_dim+d_model, K, L)
                side_info = torch.cat([ext_mask, spa_pos_emb, emb_tem_pos], dim=1)
                
                skip = []
                for layer in self.trans_layers:
                    trans_total_inp, skip_connection = layer(
                        trans_total_inp,
                        side_info,
                        emb_cond_obs,
                        emb_tem,
                        extra_spa_fea,
                        emb_diff_step
                        )
                    skip.append(skip_connection)
                    
                pred = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.trans_layers))
                pred = pred.reshape(B, self.configs.channels, K * L)
                pred = self.output_projection1(pred)  # (B,channel,K*L)
                pred = F.relu(pred)
                pred = self.output_projection2(pred)  # (B,1,K*L)
                pred = pred.reshape(B, K, L).permute(0, 2, 1) # (B, 1, K*L) -> (B, K, L) -> (B,L,K)

                coeff1 = 1 / self.alpha[s] ** 0.5
                coeff2 = (1 - self.alpha[s]) / (1 - self.alpha_hat[s]) ** 0.5

                sample = coeff1 * (sample - coeff2 * pred)

                # if then it's the last step, no need to add noise
                if s > 0: 
                    noise = torch.randn_like(sample)
                    sigma = (
                        (1.0 - self.alpha_hat[s - 1]) / (1.0 - self.alpha_hat[s]) * self.beta[s]
                    ) ** 0.5                    
                    sample += sigma * noise
            
            # use detech to create new tensor on the device, reserve sample for next iteration
            imputed_samples[:, i] = sample.detach()

        return imputed_samples
    

    def evaluate_acc(self, x_enc, x_mark_enc, mask=None, target_mask=None, weight_A=None):
        B,L,K = x_enc.shape
        imputed_samples = torch.zeros(B, self.configs.diff_samples, L, K).to(self.configs.gpu)

        for i in range(self.configs.diff_samples):
            # diffusion starts from a pure Gaussian noise
            sample = torch.randn_like(x_enc)

            # initial diffusion step start from N-1
            # if shring interval = -2, then 99, 97, 95, ... -1, 50 reverse steps
            # if shring interval = -1, then 99, 98, 97, there's no shrink
            s = self.configs.diff_steps - 1

            while True:
                if s < self.configs.sampling_shrink_interval:
                    break
                
                cond_obs = mask * x_enc
                noisy_target = target_mask * sample

                # emb_cond_obs is of shape (B, L, d_model)
                # emb_tem is of shape (B, L, d_model), timestamps embedding
                # emb_tem_pos is of shape (1, L, d_model), timestamps position embedding
                # emb_diff_step is of shape (B, diff_emb_dim)
                emb_tem, emb_tem_pos, emb_diff_step  = self.cond_embedding(cond_obs, x_mark_enc, torch.tensor([s]).to(self.configs.gpu))

                # the total input of transformer module is of shape (B,2,K,L)
                trans_total_inp = torch.cat([cond_obs.permute(0,2,1).unsqueeze(1), noisy_target.permute(0,2,1).unsqueeze(1)], dim=1)
                
                _, inputdim, K, L = trans_total_inp.shape
                
                trans_total_inp = trans_total_inp.reshape(B, inputdim, K * L)
                trans_total_inp = self.trans_inp_proj(trans_total_inp)
                trans_total_inp = F.relu(trans_total_inp)
                trans_total_inp = trans_total_inp.reshape(B, self.configs.channels, K, L)
                
                # embedding and concat side info for transformer module
                ext_mask = mask.permute(0,2,1).unsqueeze(1) # (B,1,K,L)
                spa_pos_emb = self.spa_pos_emb(
                    torch.arange(K).to(self.configs.gpu)
                    ) # (K,spa_pos_emb_dim)
                # # convert timestamp_emb from (B, L, d_model) to (B, L, K, d_model)
                # timestamp_emb = timestamp_emb.unsqueeze(2).expand(-1, -1, K, -1)
                
                # convert feature_emb from (K,spa_pos_emb_dim) to (B, L, K, spa_pos_emb_dim) -> (B, spa_pos_emb_dim, K, L)
                spa_pos_emb = spa_pos_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).permute(0,3,2,1)

                # convert emb_tem from (B, L, d_model) to (B, L, K, d_model) -> (B, d_model, K, L)
                emb_tem = emb_tem.unsqueeze(2).expand(-1, -1, K, -1).permute(0,3,2,1)
                
                # (B, K, K) -> (B, K, K, 1) -> (B, K, K, L)
                extra_spa_fea = weight_A.unsqueeze(-1).expand(-1, -1, -1, L)
                
                # emb_tem_pos is of shape (1, L, d_model) -> (1, 1, L, d_model) -> (B, K, L, d_model) -> (B, d_model, K, L)
                emb_tem_pos = emb_tem_pos.unsqueeze(0).expand(B, K, -1, -1).permute(0, 3, 1, 2)
                
                #side_info is of shape (B, 1+spa_pos_emb_dim+d_model, K, L)
                side_info = torch.cat([ext_mask, spa_pos_emb, emb_tem_pos], dim=1)
                
                skip = []
                for layer in self.trans_layers:
                    trans_total_inp, skip_connection = layer(
                        trans_total_inp,
                        side_info,
                        emb_tem,
                        extra_spa_fea,
                        emb_diff_step
                        )
                    skip.append(skip_connection)
                    
                pred = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.trans_layers))
                pred = pred.reshape(B, self.configs.channels, K * L)
                pred = self.output_projection1(pred)  # (B,channel,K*L)
                pred = F.relu(pred)
                pred = self.output_projection2(pred)  # (B,1,K*L)
                pred = pred.reshape(B, K, L).permute(0, 2, 1) # (B, 1, K*L) -> (B, K, L) -> (B,L,K)

                coeff = self.alpha_hat[s-self.configs.sampling_shrink_interval]
                sigma = (((1 - coeff) / (1 - self.alpha_hat[s])) ** 0.5) * ((1-self.alpha_hat[s] / coeff) ** 0.5)
                sample = (coeff ** 0.5) * ((sample - ((1-self.alpha_hat[s]) ** 0.5) * pred) / (self.alpha_hat[s] ** 0.5)) + ((1 - coeff - sigma ** 2) ** 0.5) * pred

                # if s == self.sampling_shrink_interval, then it's the last step, no need to add noise
                if s > self.configs.sampling_shrink_interval: 
                    noise = torch.randn_like(sample)
                    sample += sigma * noise

                s -= self.configs.sampling_shrink_interval

            imputed_samples[:, i] = sample.detach()

        return imputed_samples