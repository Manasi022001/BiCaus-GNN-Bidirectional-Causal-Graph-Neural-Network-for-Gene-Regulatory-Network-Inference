"""
Custom metrics and evaluation functions
"""
import torch
import numpy as np

def pearson_correlation(pred, target):
    """Compute Pearson correlation coefficient"""
    pred_mean = pred.mean()
    target_mean = target.mean()
    
    numerator = ((pred - pred_mean) * (target - target_mean)).sum()
    denominator = torch.sqrt(
        ((pred - pred_mean) ** 2).sum() * 
        ((target - target_mean) ** 2).sum()
    )
    return numerator / (denominator + 1e-8)

def r_squared(pred, target):
    """Compute R² score"""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return 1 - (ss_res / (ss_tot + 1e-8))
