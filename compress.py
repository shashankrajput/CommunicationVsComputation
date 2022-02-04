import os
from argparse import ArgumentParser

import torch
import torch.nn.utils.prune as prune

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


from cifar10_models.vgg import vgg11_bn
from data import CIFAR10Data
from module import CIFAR10Module







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
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="1")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    args = parser.parse_args()



    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.logger == "wandb":
        logger = WandbLogger(name=args.classifier, project="cifar10")
    elif args.logger == "tensorboard":
        logger = TensorBoardLogger("cifar10", name=args.classifier)

    
    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev + args.test_phase) else None,
        gpus=1,
        deterministic=True,
        enable_model_summary=False,
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
        precision=args.precision,
        enable_checkpointing=False
    )

    model = CIFAR10Module(args)
    data = CIFAR10Data(args)

    state_dict = os.path.join(
                "cifar10_models", "state_dicts", args.classifier + ".pt"
            )
    model.model.load_state_dict(torch.load(state_dict))

    model.model.eval() # for evaluation
    trainer=Trainer()
    
    
    parameters_to_prune=[]
    for module in model.model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    

    torch.save(model.model.state_dict(), './compressed_models/vgg11_bn_100.pt')

    sparsities=[]
    for (module, name) in parameters_to_prune:
        sparsities.append(100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()))
        
    print(100)
    print(sparsities)
    trainer.test(model, data.test_dataloader())
    
    for current in [90, 50, 10, 7.5, 5, 2, 1]:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=(100-current)/100,
        )
        sparsities=[]
        for (module, name) in parameters_to_prune:
            prune.remove(module, name)
            sparsities.append(100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement()))
        
        print(current)
        print(sparsities)
        trainer.test(model, data.test_dataloader())
        
        


        torch.save(model.model.state_dict(), './compressed_models/vgg11_bn_'+str(current)+'.pt')

    