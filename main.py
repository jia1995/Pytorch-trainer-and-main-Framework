import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import argparse
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implementation of Unet model.')
    parser.add_argument('--dataset', type=str,default='map.npz', help='dataset name.')
    parser.add_argument('--in_channels', type=int, default=1, help='Input channels number.')
    parser.add_argument('--classes', type=int, default=16, help='The number of classes.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for splitting dataset.')
    parser.add_argument('--model_path', type=str, default='models', help='Saving model path.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.99, help='Momentum of the optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='weight decay for optimizer.')
    parser.add_argument('--optim', type=str, default='Adam', help='The optimizer functions.')
    parser.add_argument('--batch_size', type=int, default=1, help='training data batch size.')
    parser.add_argument('-epochs', type=int, default=100, help='training epochs.')
    parser.add_argument('--device', type=int, default=-1, help='which gpu to use.(-1 for cpu)')
    parser.add_argument('--interval', type=int, default=10, help='interval to save the model.')
    parser.add_argument('--run_type', type=str, default='traineval', help='The running type.', choices=['traineval', 'test'])
    parser.add_argument('--lr_scheduler', type=str, default='', help='PyTorch Learning rate Adjustment Strategy.')
    parser.add_argument('--steplr_size', type=int, default=50)
    parser.add_argument('--lrs_gamma', type=float, default=0.1)
    parser.add_argument('--lr_milestones', type=list, default=[30,60])
    parser.add_argument('--model', type=str, default='', help='Pretrained model file.')
    
    args = parser.parse_args()
    dataset = MyDataset(args.dataset)
    
    model = MyModel(args.in_channels, args.classes)
    epoch = 100
    train_dataset, val_dataset, test_dataset = random_split(
                dataset=dataset,
                lengths=[56, 8, 8],
                generator=torch.Generator().manual_seed(args.seed)
            )

    dataloader1 = DataLoader(train_dataset, batch_size=args.batch_size)
    dataloader2 = DataLoader(val_dataset, batch_size=args.batch_size)
    dataloader3 = DataLoader(test_dataset, batch_size=args.batch_size)
    tin = Trainer(args, model)
    if args.run_type=='traineval':
        tin.run(dataloader1, dataloader2)
        tin.test(dataloader3)
    elif args.run_type=='test':
        tin.test(dataloader3, args.model)