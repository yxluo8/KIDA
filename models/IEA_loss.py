import torch
import torch.nn as nn
import torch.nn.functional as F

class IEALoss(nn.Module):
    """
    Intent-Execution Alignment (IEA) Loss
    Asymmetric Kinematic-Semantic regularization:
    - Normal (y=0): L1 sparsity on kinematic residual (\Delta).
    - Error (y=1): Decouple \Delta from normative intent (minimizing absolute cosine similarity).
    """
    def __init__(self, err_weight=1.25, eps=1e-8):
        super(IEALoss, self).__init__()
        self.err_weight = err_weight
        self.eps = eps

    def forward(self, delta, q_p, e_labels):
        """
        Args:
            delta (Tensor): Kinematic residual (\Delta). Shape: [B, C, L]
            q_p (Tensor): Normative intent query (Q_p). Shape: [B, C, 1]
            e_labels (Tensor): Ground truth labels. Shape: [B, L] or [L]
        Returns:
            loss_align (Tensor): Scalar IEA loss value.
        """
        L = delta.shape[2]
        
        # Expand intent query to match sequence length
        if q_p.shape[2] == 1:
            q_p_exp = q_p.repeat(1, 1, L)  # [B, C, L]
        else:
            q_p_exp = q_p
            
        # Match label dimension to batch size
        if e_labels.dim() == 1:
            e_labels = e_labels.unsqueeze(0) # [L] -> [1, L]
            
        # State-dependent Masks
        mask_normal = (e_labels == 0).float()
        mask_error  = (e_labels == 1).float()

        # Normal Path: Physical L1 Sparsity
        delta_l1 = torch.mean(torch.abs(delta), dim=1)  # [B, L]
        loss_normal = delta_l1 * mask_normal

        # Error Path: Semantic Orthogonalization (Decoupling)
        #|cos(F^\Delta, Q)|
        sim = F.cosine_similarity(delta, q_p_exp, dim=1) # [B, L]
        loss_error = torch.abs(sim) * mask_error 

        # State-wise Average
        avg_loss_normal = torch.sum(loss_normal) / (torch.sum(mask_normal) + self.eps)
        avg_loss_error  = torch.sum(loss_error) / (torch.sum(mask_error) + self.eps)

        # Asymmetric Weighting Fusion
        loss_align = avg_loss_normal + self.err_weight * avg_loss_error

        return loss_align