import os
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

class Trainer:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        if args.optim=='Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        elif args.optim == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        weight = torch.ones(16)
        weight[0] = 0.1
        weight[1:9] = 0.4
        weight[-6:] = 4
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.best_model = None
        self.best_perform = 0.
        if args.lr_scheduler=='StepLR':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.steplr_size, gamma=args.lrs_gamma)
        elif args.lr_scheduler=='MultiStepLR':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, args.milestones, args.lrs_gamma)
        elif args.lr_scheduler=='ExponentialLR':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.lrs_gamma)
        self.device = device = torch.device("cuda:" + str(args.device)) if args.device!=-1 else torch.device("cpu")

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        tot = 0.
        alls = 0
        for batch in tqdm(train_loader, desc='Iteration'):
            x, y = batch
            pred = self.model(x)
            self.optimizer.zero_grad()
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            train_loss+= loss.item()
            preds = pred.argmax(dim=1)
            preds = preds.detach().cpu()
            tot += preds.eq(y).sum()
            _,a,b = preds.size()
            alls += a*b
        print(f'train_loss: {train_loss/56:.4f}\ttrain_acc: {(tot/alls).item():.4f}')

    def eval(self, eval_loader):
        self.model.eval()
        tot, alls = 0., 0
        eval_loss = 0.
        for batch in eval_loader:
            x, y = batch
            preds = self.model(x)
            loss = self.criterion(preds, y)
            eval_loss += loss.item()
            preds = preds.argmax(dim=1)
            preds = preds.detach().cpu()
            tot += preds.eq(y).sum()
            _,a,b = preds.size()
            alls += a*b
        print(f'eval loss: {eval_loss/8:.4f}\teval acc: {(tot/alls).item():.4f}')
        return eval_loss/8, (tot/alls).item()

    def load_model(self, mname):
        self.model.load_state_dict(torch.load(f'./{self.args.model_path}/{mname}', map_location=self.device))

    def save_model(self, mname):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)
        if 'best' in mname:
            torch.save(self.best_model.state_dict(), f'./{self.args.model_path}/{mname}.pth')
        else:
            torch.save(self.model.state_dict(), f'./{self.args.model_path}/{mname}.pth')

    def run(self, tloader, eloader):
        best_e = 0
        for e in tqdm(range(self.args.epochs), desc='Epoch'):
            self.lr_scheduler.step(e)
            self.train(tloader)
            _, eacc = self.eval(eloader)
            if self.best_perform < eacc:
                self.best_model = copy.deepcopy(self.model)
                self.best_perform = eacc
                best_e = e
            if e==0 or (e+1)%self.args.interval==0:
                self.save_model(f'{e+1:0>3}_{eacc:.4f}')
        self.save_model(f'best_{best_e:0>3}_{self.best_perform:.4f}')

    def test(self, test_dataloader, mname=None):
        if mname:
            self.load_model(mname)
        _, test_acc = self.eval(test_dataloader)