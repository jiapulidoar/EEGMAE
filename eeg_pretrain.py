import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy

from config import Config_MBM_EEG
from dataset import EEGDataset
import models_mae
from engine_pretrain import train_one_epoch
from util.utils import save_model
from util.misc import NativeScalerWithGradNormCount as NativeScaler


os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init(
                    project="dreamdiffusion",
                    anonymous="allow",
                    group='stageA_sc-mbm',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training for EEG', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--blr', type=float,help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model Parameters
    parser.add_argument('--model', default='maeeg_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mask_ratio', type=float, 
                        help='Masking ratio (percentage of removed patches).') # 0.75
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--decoder_num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)

    parser.add_argument('--norm_pix_loss', type=bool, help='Use (per-patch) normalized pixels as targets for computing loss')

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)

    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)

    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local-rank', type=int)


    # For eegset 
    parser.add_argument("--data_path", type=str, default='./dataset/mne_data/', help="training data directory")
    parser.add_argument("--nfft", type=int, help="n for stft")
    parser.add_argument("--hop_length", type=int, help="hop_length for stft")
    parser.add_argument('--use_custom_patch', type=bool, default=False, help='use custom patch and override timm PatchEmbed')
    parser.add_argument('--decoder_mode', default=1, type=int,help='decoder mode 0: global attn 1: swined local attn')
    parser.add_argument('--mask_t_prob', type=float, help='ratio of masking time')
    parser.add_argument('--mask_f_prob', type=float, help='ratio of masking freq')
    parser.add_argument('--mask_2d', type=bool, help='use 2d masking')

    # set norm_pix_loss=True for normal training, norm_pix_loss=False for visualization
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)

def main(config):
    print('num of gpu:')
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    output_path = os.path.join(config.root_path, 'results', 'eeg_pretrain',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    # logger = wandb_logger(config) if config.local_rank == 0 else None
    logger = None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    spec_size=(64,64)
    in_chans = 128 
    print(config.data_path)
    dataset_train = EEGDataset(config.data_path, nfft = config.nfft, hop_length=config.hop_length, spec_size=spec_size)
   
    print(f'Dataset size: {len(dataset_train)}\n Time len: {dataset_train.data_len}')
    sampler = torch.utils.data.DistributedSampler(dataset_train, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=sampler, num_workers=10,
                shuffle=(sampler is None), pin_memory=True, drop_last=True,)

    # create model
    model = models_mae.__dict__[config.model](norm_pix_loss=config.norm_pix_loss, 	
                                            in_chans=in_chans, audio_exp=True,	
                                            img_size=spec_size,	
                                            #use_custom_patch=config.use_custom_patch,	
                                            decoder_mode=config.decoder_mode, 
                                            mask_2d=config.mask_2d, mask_t_prob=config.mask_t_prob, mask_f_prob=config.mask_f_prob)
    model.to(device)
    model_without_ddp = model

    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = config.batch_size * config.accum_iter * 1
    
    if config.lr is None:  # only base_lr is specified
        config.lr = config.blr * eff_batch_size / 256

    print("base lr: %.2e" % (config.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % config.lr)

    print("accumulate grad iterations: %d" % config.accum_iter)
    print("effective batch size: %d" % eff_batch_size)



    if torch.cuda.device_count() > 1:
        model_without_ddp = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_without_ddp)
        model_without_ddp = DistributedDataParallel(model_without_ddp, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    param_groups = optim_factory.add_weight_decay(model_without_ddp, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    cor_list = []
    start_time = time.time()
    print('Start Training the EEG MAE ... ...')
    img_feature_extractor = None
    preprocess = None
    for epoch in range(config.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            args=config
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        
        if (epoch % 20 == 0 or epoch + 1 == config.epochs) and config.local_rank == 0: #and ep != 0
            save_model(config, epoch, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,'checkpoints'))
            # plot figures
            #plot_recon_figures(model, device, dataset_pretrain, output_path, 5, config, logger, model_without_ddp)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return

@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['spec']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        # sample_with_mask = model_without_ddp.patchify(sample.transpose(1,2))[0].to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        sample_with_mask = sample.to('cpu').squeeze(0)[0].numpy().reshape(-1, model_without_ddp.patch_size)
        # pred = model_without_ddp.unpatchify(pred.transpose(1,2)).to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        # sample = sample.to('cpu').squeeze(0)[0].unsqueeze(0).numpy()
        pred = model_without_ddp.unpatchify(pred).to('cpu').squeeze(0)[0].numpy()
        # pred = model_without_ddp.unpatchify(model_without_ddp.patchify(sample.transpose(1,2))).to('cpu').squeeze(0)[0].numpy()
        sample = sample.to('cpu').squeeze(0)[0].numpy()
        mask = mask.to('cpu').numpy().reshape(-1)

        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)


def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_EEG()
    config = update_config(args, config)
    main(config)
    