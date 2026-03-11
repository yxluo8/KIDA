import torch
import torch.nn as nn
import copy
from mamba_ssm import Mamba
from timm.models.layers import DropPath

from .KEB import KEB
from .CIB import CIB

class SingleStageModel(nn.Module):
    # MTemp: Mamba-based Temporal Module
    def __init__(self, com_factor, dim, out_f_maps):
        super(SingleStageModel, self).__init__()
        base_channels = com_factor // 8

        self.conv_1x1 = nn.Conv1d(dim, com_factor, 1)
        self.conv_out = nn.Conv1d(com_factor, out_f_maps, 1)

        self.conv1 = nn.Conv1d(com_factor, base_channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4)
        self.conv3 = nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=8, dilation=8)
        self.conv_1x1_output = nn.Conv1d(com_factor + base_channels * 3, com_factor, 1)
        
        self.drop_path = DropPath(0.1)
        self.layers = nn.ModuleList([Mamba(d_model=com_factor, d_state=16, d_conv=4, expand=2) for i in range(1)])

    def forward(self, x):
        out = self.conv_1x1(x)

        x1 = self.conv1(out)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = torch.cat([out, x1, x2, x3], dim=1)
        out = self.conv_1x1_output(out)

        out = out.permute(0, 2, 1)  
        for i, layer in enumerate(self.layers):
            shortcut = out
            out = layer(out)
            out = shortcut + self.drop_path(out)  
                
        out = self.conv_out(out.permute(0, 2, 1))
        return out


class MultiStageModel(nn.Module):
    # KIDA Framework
    def __init__(self, num_block, et, com_factor, text_d, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.emb_type = et
        
        self.keb = KEB(flow_dim=6, dim=dim)
        self.cib = CIB(text_d=text_d, dim=dim)

        self.stages = nn.ModuleList([
            copy.deepcopy(SingleStageModel(com_factor, int(dim // (2**s)), int(dim // (2 ** (s + 1))))) 
            for s in range(num_block)
        ])
        
        self.classify = nn.Conv1d(int(dim / (2 ** (num_block))), num_classes, 1)
        self.out_proj = nn.Linear(dim, 1)   
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, text_pfe, flow_mag=None):
        if x.size(1) != 1000:
            x = x.permute(0, 2, 1)
        
        # 1. Kinematic Execution Branch (KEB)
        x, delta, f_delta = self.keb(x, flow_mag)

        # 2. Clinically-Guided Intent Branch (CIB)
        x, f_text_p = self.cib(x, text_pfe)

        if self.training:
            sigma = 0.05   
            noise = torch.randn_like(x) * sigma
            x = x + noise

        # 3. Mamba-based Temporal Module (MTemp)
        for i, s in enumerate(self.stages):
            x = s(x)
            # Inject \Delta at the first temporal layer
            if i == 0:
                delta = s(delta)
                x = x + delta
                
        x = self.classify(x)

        return x, f_delta, f_text_p