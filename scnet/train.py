import os
import torch
from torch.utils.data import DataLoader
from .wav import get_wav_datasets
from .SCNet import SCNet
from .solver import Solver
import argparse
import yaml
from ml_collections import ConfigDict
from accelerate import Accelerator
from .log import logger

accelerator = Accelerator()

def get_model(config):

    model = SCNet(**config.model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters: {total_params}")
    return model


def get_solver(args):
    with open(args.config_path, 'r') as file:
          config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

    torch.manual_seed(config.seed)
    model = get_model(config)
  
    # torch also initialize cuda seed if available
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if config.optim.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)
    elif config.optim.optim == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.optim.lr,
            betas=(config.optim.optim.momentum, config.optim.beta2),
            weight_decay=config.optim.weight_decay)

    train_set, valid_set = get_wav_datasets(config.data)

    logger.info("train/valid set size: %d %d", len(train_set), len(valid_set))
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.misc.num_workers, drop_last=True)
    train_loader = accelerator.prepare_data_loader(train_loader)

    valid_loader = DataLoader(
        valid_set, batch_size=1, shuffle=False,
        num_workers=config.misc.num_workers)
    valid_loader = accelerator.prepare_data_loader(valid_loader)

    loaders = {"train": train_loader, "valid": valid_loader}

    model, optimizer = accelerator.prepare(model, optimizer)
    
    return Solver(loaders, model, optimizer, config, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='./result/', help="path to config file")
    parser.add_argument("--config_path", type=str, default='./conf/config.yaml', help="path to save checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if not os.path.isfile(args.config_path):
        print(f"Error: config file {args.config_path} does not exist.")
        sys.exit(1)
        
    solver = get_solver(args)
    accelerator.wait_for_everyone()
    solver.train()



if __name__ == "__main__":
    main()

