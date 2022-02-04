import os
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module

def main(args):

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier+args.compressedModel, project="cifar10")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("cifar10_logs", name=args.classifier+args.compressedModel)


    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev + args.test_phase) else None,
        gpus=1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        enable_checkpointing=False,
        precision=args.precision
    )



    model = CIFAR10Module(args)
    data = CIFAR10Data(args)

   
    state_dict = os.path.join(
        "compressed_models", "vgg11_bn_"+args.compressedModel + ".pt"
    )
    model.model.load_state_dict(torch.load(state_dict))

    model.train_dataloader=data.train_dataloader

    trainer.validate(model, data)
    trainer.fit(model, data)
    # trainer.fit(model, data.train_dataloader, data.test_dataloader)
    # trainer.test(model, data.test_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="vgg11_bn")
    parser.add_argument("--compressedModel", type=str, default='5')

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="1")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()

    for compression in [100, 90, 50, 10, 7.5, 5, 2, 1]:
        args.compressedModel=str(compression)
        main(args)
