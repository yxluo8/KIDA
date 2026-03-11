import torch
import torch.nn as nn

class CIB(nn.Module):
    """
    Clinically-Guided Intent Branch (CIB)
    Formulates normative intent (Q_p) and aligns visual representations via cross-attention.
    """
    def __init__(self, text_d, dim):
        super().__init__()
        # Normative intent queries
        self.lq = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, text_d // 2)) for _ in range(2)])
        self.pre_conv = nn.Conv1d(text_d + text_d // 2, dim, 1)
        
        # Projection for Intent-Execution Alignment (IEA) Loss
        self.proj_dim = 256
        self.proj_text = nn.Sequential(
            nn.Conv1d(text_d, 512, 1),
            nn.GELU(),
            nn.Conv1d(512, self.proj_dim, 1)
        )
        
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, text_pfe):
        # Formulate intent query (Q_p)
        q_p = torch.cat([self.lq[0], text_pfe], dim=2)
        q_p = self.pre_conv(q_p.permute(0, 2, 1)) # [B, dim, 1]
        
        # Project semantic intent for IEA regularization
        f_text_p = self.proj_text(text_pfe.permute(0, 2, 1)) # [B, 256, 1]
        
        # Cross-attention alignment
        k = x.permute(0, 2, 1) 
        v = k
        attn_p, _ = self.cross_attn(q_p.permute(0, 2, 1), k, v)  
        attn_out = attn_p.permute(0, 2, 1)
        
        # Semantic-aware visual representations
        x_aligned = x + self.alpha * attn_out  
        
        return x_aligned, f_text_p