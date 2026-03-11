import torch
import torch.nn as nn

class MotionAdapter(nn.Module):
    # KEB: Feature Fusion Module
    def __init__(self, flow_dim=6, embed_dim=1000):
        super().__init__()
        self.flow_embedding = nn.Sequential(
            nn.Conv1d(flow_dim, 125, kernel_size=1),
            nn.GroupNorm(5, 125), 
            nn.GELU(),
            nn.Conv1d(125, embed_dim, kernel_size=1),
            nn.GroupNorm(50, embed_dim),
            nn.GELU()
        )
        self.flow_encoder = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim), 
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1), 
            nn.Sigmoid() 
        )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, vis_feat, flow_feat):
        if flow_feat.device != vis_feat.device:
            flow_feat = flow_feat.to(vis_feat.device)
        flow_emb = self.flow_embedding(flow_feat)
        motion_mask = self.flow_encoder(flow_emb)
        flow_res = self.alpha * (flow_emb * motion_mask)
        out = vis_feat + flow_res
        return out, flow_res

class KEB(nn.Module):
    # Kinematic Execution Branch (KEB)
    def __init__(self, flow_dim, dim):
        super().__init__()
        self.MotionAdapter = MotionAdapter(flow_dim=flow_dim, embed_dim=dim)
        
        # Extractor for kinematic residual (\Delta)
        self.motion_to_shift = nn.Conv1d(flow_dim, dim, 1)
        
        # Projection for IEA regularization
        self.proj_dim = 256
        self.proj_delta = nn.Sequential(
            nn.Conv1d(dim, 512, 1),
            nn.GELU(),
            nn.Conv1d(512, self.proj_dim, 1)
        )

    def forward(self, x, flow_mag):
        flow_res = None
        if flow_mag is not None:
            if flow_mag.shape[1] != 6:
                flow_mag = flow_mag.permute(0, 2, 1)
            x, flow_res = self.MotionAdapter(x, flow_mag)

        # delta: \Delta
        f_delta = None
        delta = self.motion_to_shift(flow_mag) 
        
        # f_delta: Projected \Delta for IEA
        if flow_res is not None:
            f_delta = self.proj_delta(delta) 
            
        return x, delta, f_delta