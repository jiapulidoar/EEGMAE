import numpy as np
import math
import torch
import os


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < config.warmup_epochs:
        lr = config.lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (config.lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (config.num_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_model(config, epoch, model, optimizer, loss_scaler, checkpoint_paths):
    os.makedirs(checkpoint_paths, exist_ok=True)
    to_save = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict(),
        'config': config,
    }
    torch.save(to_save, os.path.join(checkpoint_paths, 'checkpoint.pth'))
    

def load_model(config, model, checkpoint_path ):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(f'Model loaded with {checkpoint_path}')
