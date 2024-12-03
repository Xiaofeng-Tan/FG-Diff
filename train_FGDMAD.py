import argparse
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from models.FGDMAD import FGDMAD
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader
from utils.ema import EMACallback
from pytorch_lightning.loggers import TensorBoardLogger
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__== '__main__':

    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='Pose_AD_Experiment')
    parser.add_argument('-c', '--config', type=str, required=True,
                        default='/your_default_config_file_path')

    args = parser.parse_args()
    config_path = args.config
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    args = init_args(args)
    # Save config file to ckpt_dir
    os.system(f'cp {config_path} {os.path.join(args.ckpt_dir, "config.yaml")}')     
    
    # Set seeds    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 
    pl.seed_everything(args.seed)

    # Set callbacks and logger
    if (hasattr(args, 'diffusion_on_latent') and args.stage == 'pretrain'):
        monitored_metric = 'pretrain_rec_loss'
        metric_mode = 'min'
    elif args.validation:
        monitored_metric = 'AUC'
        metric_mode = 'max'
    else:
        monitored_metric = 'loss_noise'
        metric_mode = 'min'
    callbacks = [ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=3,
                                 monitor=monitored_metric,
                                 mode=metric_mode, auto_insert_metric_name=True)]
    
    callbacks += [EMACallback()] if args.use_ema else []
    
    if args.use_wandb:
        callbacks += [LearningRateMonitor(logging_interval='step')]
        wandb_logger = WandbLogger(project=args.project_name, group=args.group_name, entity=args.wandb_entity, 
                                   name=args.dir_name, config=vars(args), log_model='all')
    else:
        wandb_logger = False

    # Get dataset and loaders
    #args.split = 'test'
    #dataset, loader, _, _ = get_dataset_and_loader(args, split=args.split)
    #args.split = 'train'
    _, train_loader, _, val_loader = get_dataset_and_loader(args, split=args.split,validation=args.validation)

    #exit()
    # Initialize model and trainer
    model = FGDMAD(args)
    
    logger = TensorBoardLogger(save_dir='/root/tf-logs/', name=args.dataset_choice+'_new')

    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, default_root_dir=args.ckpt_dir, max_epochs=args.n_epochs, 
                         logger=logger, callbacks=callbacks, strategy=DDPStrategy(find_unused_parameters=False), 
                         log_every_n_steps=20, num_sanity_val_steps=0)

    # Train the model    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.save_checkpoint(args.ckpt_dir + "/checkpoint_ae.ckpt")
    
    out = trainer.test(model, dataloaders=val_loader)
