import argparse
import os

import pytorch_lightning as pl
import yaml
from models.FGDMAD import FGDMAD
from utils.argparser import init_args
from utils.dataset import get_dataset_and_loader
import torch
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger

if __name__== '__main__':
    
    # Parse command line arguments and load config file
    parser = argparse.ArgumentParser(description='MoCoDAD')
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    args = argparse.Namespace(**args)
    args = init_args(args)
    # Set seeds    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed) 
    pl.seed_everything(args.seed)
    #torch.cuda.manual_seed(args.seed)
    
    logger = TensorBoardLogger(save_dir='/root/tf-logs/', name="my_model")

    # Initialize the model
    model = FGDMAD(args)
    
    if args.load_tensors:
        # Load tensors and test
        model.test_on_saved_tensors(split_name=args.split)
    else:
        # Load test data
        print('Loading data and creating loaders.....')
        ckpt_path = os.path.join(args.ckpt_dir, args.load_ckpt)
        dataset, loader, _, _ = get_dataset_and_loader(args, split=args.split)
        #_, data_loader, _, val_loader = get_dataset_and_loader(args, split=args.split, validation=args.validation)
        # Initialize trainer and test
        trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices[:1],
                             default_root_dir=args.ckpt_dir, max_epochs=1, logger=logger)
        out = trainer.test(model, dataloaders=loader, ckpt_path=ckpt_path)