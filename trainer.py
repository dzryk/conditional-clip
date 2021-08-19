import os
import torch
import numpy as np
import pytorch_lightning as pl

from CLIP import clip
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import dataset
import model


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_datadir', type=str)
    parser.add_argument('--dev_datadir', type=str)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--ft', type=str, default='bias')
    parser.add_argument('--web_dataset', type=bool, default=False)
    parser.add_argument('--wds_keys', type=str, default='img,cap')
    parser.add_argument('--dataset_size', nargs='+', type=int, default=[1e9])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--val_after_n_epochs', type=int, default=1)
    parser.add_argument('--tmax', type=int, default=1e5)
    parser.add_argument('--save_top_k', type=int, default=10)
    parser.add_argument('--num_sanity_val_steps', type=int, default=0)
    parser.add_argument('--lrate', type=float, default=3e-4)
    args = parser.parse_args()

    # Set wandb cache dir
    os.environ['WANDB_CACHE_DIR'] = args.logdir

    # Determine world size
    if args.tpu_cores:       
        import torch_xla.core.xla_model as xm
        args.world_size = xm.xrt_world_size()
    else: 
        if args.gpus > 1:     
            torch.distributed.init_process_group(backend='nccl') 
            args.world_size = torch.distributed.get_world_size()
        else:
            args.world_size = 1

    # Train model
    wandb_logger = WandbLogger(
        save_dir=args.logdir,
        project='conditional-clip',
        log_model='all')
    ckpt_callback = ModelCheckpoint(
        monitor='vloss',
        mode='min',
        filename='-{epoch:02d}-{vloss:.3f}',
        save_top_k=args.save_top_k)
    datamodule = dataset.DataModule(
        train_datadir=args.train_datadir,
        dev_datadir=args.dev_datadir,
        batch_size=args.batch_size,
        web_dataset=args.web_dataset,
        wds_keys=args.wds_keys,
        world_size=args.world_size,
        dataset_size=args.dataset_size,
        nworkers=args.nworkers)
    net = model.Model(args)
    trainer = pl.Trainer(
        default_root_dir=args.logdir,
        logger=wandb_logger,
        gpus=args.gpus,
        precision=args.precision,
        tpu_cores=args.tpu_cores,
        max_steps=args.tmax,
        num_sanity_val_steps=args.num_sanity_val_steps,
        callbacks=[ckpt_callback],
        check_val_every_n_epoch=args.val_after_n_epochs)
    trainer.fit(net, datamodule)


if __name__ == '__main__':
    main()
