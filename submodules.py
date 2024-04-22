import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,kernel_size=3,
            padding=padding, padding_mode='circular', bias=False
            )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()
    
class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='5min'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        five_min_size = 288
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        elif freq == '5min':
            self.minute_embed = Embed(five_min_size, d_model)
        
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x
    
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # see features_by_offsets in utils.py
        freq_map = {'h': 4, 't': 4, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, '5min': 5}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        # square bracket indexing 
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, dim, 1) / dim
        ) 
        div_term = div_term.unsqueeze(0)  # (1,dim)
        table = steps * div_term  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class CondObsEmbedding(nn.Module):
    """
    This is the Embedding layer only for Conditional Observation
    """
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1, diff_steps=100, diff_emb_dim=32):
        super(CondObsEmbedding, self).__init__()

        self.cond_value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # embedding layer for diffusion step
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=diff_emb_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, cond_obs, x_mark, diff_step):
        # (B,) -> (B, d_model) -> (B, d_model)
        diff_step_embedding = self.diffusion_embedding(diff_step)

        
        pos_embedding = self.position_embedding(cond_obs)

        tem_embedding = self.temporal_embedding(x_mark)
        
        if x_mark is None:
            # self.value_embedding(x) is of shape (B, L_hist, d_model)
            # fea_embedding is of shape (B, L_hist, d_model)
            # self.position_embedding(x) is of shape (1, L_hist, d_model)
            x = self.cond_value_embedding(cond_obs) + pos_embedding
        else:
            x = self.cond_value_embedding(cond_obs) + tem_embedding + pos_embedding
        
        return self.dropout(x), tem_embedding, pos_embedding, diff_step_embedding

class SpaObsEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='timeF', freq='h', dropout=0.1, diff_steps=100, diff_emb_dim=32):
        super(SpaObsEmbedding, self).__init__()
    
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        # embedding layer for diffusion step
        self.diffusion_embedding = DiffusionEmbedding(num_steps=diff_steps, embedding_dim=diff_emb_dim)

    def forward(self, cond_obs, x_mark, diff_step):
        # (B,) -> (B, d_model) -> (B, d_model)
        diff_step_embedding = self.diffusion_embedding(diff_step)
        
        pos_embedding = self.position_embedding(cond_obs)

        tem_embedding = self.temporal_embedding(x_mark)
        
        return tem_embedding, pos_embedding, diff_step_embedding

class VariEmbedding(nn.Module):
    """
    This is the Embedding layer only for Conditional Observation
    """
    def __init__(self, configs):
        super(VariEmbedding, self).__init__()
        embed_type = configs.embed
        self.value_embedding = Conv1d_with_init(1, configs.trans_channels, 1)
        self.position_embedding = PositionalEmbedding(d_model=configs.d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=configs.d_model, embed_type=embed_type,
                                                    freq=configs.freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=configs.d_model, embed_type=embed_type, freq=configs.freq)
                                                    
        # embedding for features, i.e., different locations
        self.spa_pos_emb = nn.Embedding(configs.enc_in, configs.spa_pos_emb_dim)
        self.configs  = configs

    def forward(self, inp, x_mark):
        
        B, L, K = inp.shape
        # (B, L, K) -> (B, 1, L, K) -> (B, 1, L*K) -> (B, d_model, L*K)
        inp_emb = self.value_embedding(inp.unsqueeze(1).reshape(B, 1, L*K))
        inp_emb = F.relu(inp_emb)
        inp_emb = inp_emb.reshape(B, self.configs.trans_channels, K, L) # (B, d_model, K, L)
        
        spa_pos_emb = self.spa_pos_emb(
            torch.arange(K).to(self.configs.gpu)
            ) # (K,spa_pos_emb_dim))
        # convert feature_emb from (K,spa_pos_emb_dim) to (B, L, K, spa_pos_emb_dim) -> (B, spa_pos_emb_dim, K, L)
        spa_pos_emb = spa_pos_emb.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1).permute(0,3,2,1)

        tem_pos_emb = self.position_embedding(inp)
        # emb_tem_pos is of shape (1, L, d_model) -> (1, 1, L, d_model) -> (B, K, L, d_model) -> (B, d_model, K, L)
        tem_pos_emb = tem_pos_emb.unsqueeze(0).expand(B, K, -1, -1).permute(0, 3, 1, 2)

        tem_emb = self.temporal_embedding(x_mark)
        # convert tem_embedding from (B, L, d_model) to (B, L, K, d_model) -> (B, d_model, K, L)
        tem_emb = tem_emb.unsqueeze(2).expand(-1, -1, K, -1).permute(0,3,2,1)
        
        
        return inp_emb, tem_emb, tem_pos_emb, spa_pos_emb

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=1*i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                            num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                            num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if self.seq_len % period != 0:
                length = (
                                 (self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).cuda()
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                            N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            B, C, num_period, period = out.shape
            # out = out.squeeze(2) # (B, C, L)
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])

        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=32, dropout=0.1, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, fusion_d, diffusion_embedding_dim, nheads, configs):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.fusion_projection = Conv1d_with_init(fusion_d, channels, 1)
        self.timesnet = TimesBlock(configs)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.tem_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.spa_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.tem_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_spatial(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.spa_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_info, emb_cond_obs, emb_tem_fea, emb_spa_fea, diff_step_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diff_step_emb).unsqueeze(-1)  # (B,channel,1)
        
        emb_cond_obs = self.layer_norm(self.timesnet(emb_cond_obs))
            
        # convert emb_cond_obs from (B, L, d_model) to (B, L, K, d_model) -> (B, d_model, K, L)
        emb_cond_obs = emb_cond_obs.unsqueeze(2).expand(-1, -1, K, -1).permute(0,3,2,1)
            
        emb_cond_obs = emb_cond_obs.reshape(B, -1, K * L) # (B,tn_d_model,K*L)
        emb_tem_fea = emb_tem_fea.reshape(B, -1, K * L) # (B,tn_d_model,K*L)
        emb_spa_fea = emb_spa_fea.reshape(B, -1, K * L) # (B, K, K*L)
        
        y = x + diffusion_emb
        y = torch.cat([y, emb_tem_fea, emb_spa_fea], dim=1) # (B,channels+tn_d_model+K,K*L) fusion_d = channels+tn_d_model+K
        y = self.fusion_projection(y)  # (B,channel,K*L)
        
        y = self.forward_time(y, base_shape)
        y = self.forward_spatial(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = torch.cat([side_info, emb_cond_obs], dim=1)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    
class TemResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, fusion_d, diffusion_embedding_dim, nheads, configs):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.fusion_projection = Conv1d_with_init(fusion_d, channels, 1)
        self.timesnet = TimesBlock(configs)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.tem_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.layer_norm = nn.LayerNorm(configs.d_model)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.tem_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_info, emb_cond_obs, emb_tem_fea, emb_spa_fea, diff_step_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diff_step_emb).unsqueeze(-1)  # (B,channel,1)
        
        emb_cond_obs = self.layer_norm(self.timesnet(emb_cond_obs))
            
        # convert emb_cond_obs from (B, L, d_model) to (B, L, K, d_model) -> (B, d_model, K, L)
        emb_cond_obs = emb_cond_obs.unsqueeze(2).expand(-1, -1, K, -1).permute(0,3,2,1)
            
        emb_cond_obs = emb_cond_obs.reshape(B, -1, K * L) # (B,tn_d_model,K*L)
        emb_tem_fea = emb_tem_fea.reshape(B, -1, K * L) # (B,tn_d_model,K*L)
        emb_spa_fea = emb_spa_fea.reshape(B, -1, K * L) # (B, K, K*L)
        
        y = x + diffusion_emb
        y = torch.cat([y, emb_tem_fea, emb_spa_fea], dim=1) # (B,channels+tn_d_model+K,K*L) fusion_d = channels+tn_d_model+K
        y = self.fusion_projection(y)  # (B,channel,K*L)
        y = self.forward_time(y, base_shape)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = torch.cat([side_info, emb_cond_obs], dim=1)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

class SpaResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, fusion_d, diffusion_embedding_dim, nheads, configs):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.fusion_projection = Conv1d_with_init(fusion_d, channels, 1)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.spa_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_spatial(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.spa_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_info, emb_tem_fea, emb_spa_fea, diff_step_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diff_step_emb).unsqueeze(-1)  # (B,channel,1)
        
        emb_tem_fea = emb_tem_fea.reshape(B, -1, K * L) # (B,tn_d_model,K*L)
        emb_spa_fea = emb_spa_fea.reshape(B, -1, K * L) # (B, K, K*L)
        
        y = x + diffusion_emb
        y = torch.cat([y, emb_tem_fea, emb_spa_fea], dim=1) # (B,channels+tn_d_model+K,K*L) fusion_d = channels+tn_d_model+K
        y = self.fusion_projection(y)  # (B,channel,K*L)
        
        y = self.forward_spatial(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        # side_info = torch.cat([side_info, emb_cond_obs], dim=1)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
    
class TranResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, fusion_d, nheads, configs):
        super().__init__()
        self.fusion_projection = Conv1d_with_init(fusion_d, channels, 1)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, side_info, emb_tem_fea, emb_spa_fea):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        emb_tem_fea = emb_tem_fea.reshape(B, -1, K * L) # (B,tn_d_model,K*L)
        emb_spa_fea = emb_spa_fea.reshape(B, -1, K * L) # (B, K, K*L)
        
        y = torch.cat([x, emb_tem_fea, emb_spa_fea], dim=1) # (B,channels+tn_d_model+K,K*L) fusion_d = channels+tn_d_model+K
        y = self.fusion_projection(y)  # (B,channel,K*L)
        
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, side_dim, _, _ = side_info.shape
        side_info = side_info.reshape(B, side_dim, K * L)
        side_info = self.cond_projection(side_info)  # (B,2*channel,K*L)
        y = y + side_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip