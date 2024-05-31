import os
import numpy as np

class Config_MAE_fMRI: # back compatibility
    pass
class Config_MBM_finetune: # back compatibility
    pass 

class Config_MBM_EEG(Config_MAE_fMRI):
    # configs for fmri_pretrain.py
    def __init__(self):
    # --------------------------------------------
        # Training Parameters
        self.lr = None#2.5e-4
        self.min_lr = 0.
        self.blr = 1e-3
        self.weight_decay = 0.05
        self.epochs = 30
        self.batch_size = 64
        self.clip_grad = 0.8
        self.warmup_epochs = 3
        
        # Model Parameters
        self.model = 'maeeg_vit_base_patch16'
        self.mask_ratio = 0.6
        self.patch_size = 4 #  1
        self.embed_dim = 32 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 128 #128
        self.depth = 6
        self.num_heads = 8
        self.decoder_num_heads = 8
        self.mlp_ratio = 1.0
        self.use_custom_patch= False 
        self.decoder_mode=1
        self.mask_t_prob=0.3
        self.mask_f_prob=0.3
        self.mask_2d=False 

        # Project setting
        self.root_path = '../EEGMAE/'
        self.output_dir = '../EEGMAE/results/'
        self.seed = 42
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6
        self.resume = ''

        # distributed training
        self.local_rank = 0

        #EEG 
        self.data_path = './dataset/physionet/30s_splited'
        self.nfft = 128
        self.hop_length= 64
        self.norm_pix_loss=False 




class Config_EEG_finetune(Config_MBM_finetune):
    def __init__(self):
        
        self.lr = None#2.5e-4
        self.min_lr = 0.
        self.blr = 1e-3
        self.weight_decay = 0.05
        self.epochs = 30
        self.batch_size = 128
        self.clip_grad = 0.8
        self.warmup_epochs = 3
        
        # Model Parameters
        self.model = 'maeeg_vit_base_patch16'
        self.mask_ratio = 0.3
        self.patch_size = 4 #  1
        self.embed_dim = 1024 #256 # has to be a multiple of num_heads
        self.decoder_embed_dim = 512 #128
        self.num_heads = 16
        self.decoder_num_heads = 16
        self.mlp_ratio = 1.0
        self.use_custom_patch= False 
        self.decoder_mode=1
        self.mask_t_prob=0.3
        self.mask_f_prob=0.3
        self.mask_2d=True 

        # Project setting
        self.root_path = '../EEGMAE/'
        self.output_dir = '../EEGMAE/results/'
        self.seed = 42
        self.roi = 'VC'
        self.aug_times = 1
        self.num_sub_limit = None
        self.include_hcp = True
        self.include_kam = True
        self.accum_iter = 1

        self.use_nature_img_loss = False
        self.img_recon_weight = 0.5
        self.focus_range = None # [0, 1500] # None to disable it
        self.focus_rate = 0.6
        self.resume = ''

        # distributed training
        self.local_rank = 0

        #EEG 
        self.data_path = './dataset/physionet/30s_splited'
        self.nfft = 128
        self.hop_length= 64
        self.norm_pix_loss=False 

        
class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '../dreamdiffusion/'
        self.output_path = '../dreamdiffusion/exps/'

        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.clip_tune = True #False
        self.cls_tune = False
        self.subject = 4
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 



class Config_Cls_Model:
    def __init__(self):
        # project parameters
        self.seed = 2022
        self.root_path = '../dreamdiffusion/'
        self.output_path = '../dreamdiffusion/exps/'

        # self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_5_95_std.pth')
        self.eeg_signals_path = os.path.join(self.root_path, 'datasets/eeg_14_70_std.pth')
        # self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_single.pth')
        self.splits_path = os.path.join(self.root_path, 'datasets/block_splits_by_image_all.pth')
        self.roi = 'VC'
        self.patch_size = 4 # 16
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.mlp_ratio = 1.0

        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains')
        
        self.dataset = 'EEG' 
        self.pretrain_mbm_path = None

        self.img_size = 512

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 5 if self.dataset == 'GOD' else 25
        self.lr = 5.3e-5
        self.num_epoch = 50
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.15
        self.global_pool = False
        self.use_time_cond = False
        self.clip_tune = False
        self.subject = 4
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None 