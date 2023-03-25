import argparse

from loader import MoleculeDataset
from loader import graph_data_obj_to_mol_simple

from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_scaffold_split, random_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
from notify import send_msg
from torch.distributions.dirichlet import Dirichlet
criterion = nn.BCEWithLogitsLoss(reduction = "none")
from rdkit import Chem

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss= 0 #float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.best_test_loss = 0
        
    def __call__(
        self, current_valid_loss, current_test_loss,
        epoch, model, optimizer, criterion, fname
    ):
        print(f'Best valid: {self.best_valid_loss} Best test: {self.best_test_loss} current : {current_valid_loss} ')
        if current_valid_loss > self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_test_loss = current_test_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{fname}/best_model.pth')

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        if not args.freeze_model:
            loss.backward()

        optimizer.step()


def eval(args, model, device, loader, fname):
    model.eval()
    y_true = []
    y_scores = []
    smiles = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        for dt in batch.to_data_list():
            mol=graph_data_obj_to_mol_simple(dt.x, dt.edge_index, dt.edge_attr)
            smiles.append(Chem.MolToSmiles(mol))
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    probs = torch.sign(torch.from_numpy(y_scores)).numpy()
    roc_list = []
    os.makedirs(fname, exist_ok=True)
    with open(fname + '/errorcase', 'w') as output_file:
        for i in range(len(y_true)):
            output_file.write(str(y_true[i]) + ' ')
            output_file.write(str(probs[i]) + ' ')
            output_file.write(str(probs[i]==y_true[i]) + ' ')
            # output_file.write(str(y_scores[i]) + ' ')
            output_file.write(smiles[i] + '\n')
    
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))
    # print(roc_list) # task 끼리의 ROC 평균내서 성능측정
    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def model_recycling(model, averaging_target, averaging_aux ,model_ver, model_weight):
    checkpoints=[]
    for aux in averaging_aux:
        print(aux)
        if model_ver.endswith("nfr"):
            checkpoints.append(torch.load(f'/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/non_freeze_bn/target_{averaging_target}_aux_{aux}_{model_ver}_500/best_model.pth')['model_state_dict'])
        else:
            checkpoints.append(torch.load(f'/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/freeze_bn/target_{averaging_target}_aux_{aux}_{model_ver}_500/best_model.pth')['model_state_dict'])
    #target model 
    # checkpoints.append(torch.load(f'/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/freeze_bn/{averaging_target}_500/best_model.pth')['model_state_dict'])
    checkpoints.append(torch.load(f'/disk/bubble3jh/DomainBed/pretrain-gnns/chem/runs/finetune_cls_runseed0/0215exp/_{averaging_target}_fbn{averaging_target}_lpft_500/best_model.pth')['model_state_dict'])
    temp = dict.fromkeys(checkpoints[0].keys(),0)
    
    for i, checkpoint in enumerate(checkpoints):
        for key in checkpoint:
            temp[key] = temp[key] + model_weight[i]*checkpoint[key]
    model.load_state_dict(temp)

softmax= torch.nn.Softmax(dim=0)

def set_seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', action= 'store_true', help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--replace_classifier', default='', type=str)   
    parser.add_argument('--averaging_target', default='', type=str)   
    parser.add_argument('--averaging_aux', type=str, nargs='+') 
    parser.add_argument('--z_tensor', type=float, nargs='+') 
    parser.add_argument('--freeze_gnn', action= 'store_true', help='freeze featurizer')
    parser.add_argument('--freeze_lc', action= 'store_true', help='freeze lc')
    parser.add_argument('--freeze_bn', action= 'store_true', help='freeze bn')
    parser.add_argument('--freeze_model', action= 'store_true', help='freeze model')
    parser.add_argument('--model_ver', default='', type=str)   
    parser.add_argument('--ensemble_method', default='average', type=str) 
    model_weight=0
    args = parser.parse_args()
    save_best_model = SaveBestModel()
    set_seed_all(args.runseed)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    
    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba": # 제외
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[1])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    
    # mol=graph_data_obj_to_mol_simple(test_loader.dataset[0].x, test_loader.dataset[0].edge_index, test_loader.dataset[0].edge_attr)
    # smiles=Chem.MolToSmiles(mol)
    # print(smiles)
    # print(test_loader.dataset[0].x.shape)
    # print(test_loader.dataset[0].edge_index.shape)
    # print(test_loader.dataset[0].edge_attr.shape)
    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)

    if not args.input_model_file == "":
        if not args.replace_classifier == "":
            checkpoint = torch.load(args.input_model_file)['model_state_dict']
            ft_pt = {key.replace('gnn.', ''): checkpoint.copy().pop(key) for key in checkpoint.keys()}
            gnn_pt = {key: value for key, value in ft_pt.copy().items() if key != "graph_pred_linear.weight" and key != "graph_pred_linear.bias"}
            lp_checkpoint = torch.load(args.replace_classifier)
            lp_pt = {key.replace('graph_pred_linear.', ''): lp_checkpoint.copy().pop(key) for key in lp_checkpoint.keys()}
            lnr_pt = {key: value for key, value in lp_pt.copy().items() if key == "weight" or key == "bias"}
            
            model.gnn.load_state_dict(gnn_pt)
            model.graph_pred_linear.load_state_dict(lnr_pt)
        else:
            # model.from_pretrained(args.input_model_file) #first finetuning
            model.load_state_dict(torch.load(args.input_model_file)) #best model
    if not args.averaging_target == "":
        if args.ensemble_method=="average":
            model_z = torch.ones(len(args.averaging_aux)+1, requires_grad = False)
            model_weight = softmax(model_z)
        elif args.z_tensor == None:
            d = Dirichlet(torch.ones(len(args.averaging_aux)+1))
            torch.random.seed()
            model_weight = d.sample()
            torch.manual_seed(args.runseed)
        else:
            model_weight = torch.tensor(args.z_tensor)
            print('model_weight : ')
            print(model_weight)
        model_recycling(model, args.averaging_target, args.averaging_aux ,args.model_ver, model_weight)

    if args.freeze_bn:
        for param in model.gnn.batch_norms.parameters():
            param.requires_grad = False    

    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    if not args.freeze_gnn:
        model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    # if not args.ensemble_method == "average":
    if not args.averaging_target == "":
        model_param_group.append({"params": model_weight})
    if not args.freeze_lc:
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    
    if args.freeze_bn and args.replace_classifier=="":
        args.filename = "_fbn" + args.filename
    if args.freeze_gnn:
        args.filename = "_fgnn" + args.filename
    if args.freeze_lc:
        args.filename = "_flc" + args.filename 
    if not args.model_ver == "":
        args.filename = "_" + args.model_ver  + args.filename
    if args.replace_classifier=="":
        args.filename = "_" +args.dataset  +args.filename
    if args.averaging_target:
        args.filename =  "recycling" + args.filename 

    fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/'+ "errorcase/" + args.filename + "_" + str(args.epochs)
    if not args.ensemble_method=="average":
        fname = fname + "_" + str(model_weight[0].item())
    print(f"saving model to : [{fname}]")
    #delete the directory if there exists one
    if os.path.exists(fname):
        shutil.rmtree(fname)
        print("removed the existing file.")
    writer = SummaryWriter(fname)
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        if args.freeze_model:
            with torch.no_grad():
                train(args, model, device, train_loader, optimizer)
            # print("freezed")
        else:
            print("training")
            train(args, model, device, train_loader, optimizer)
        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, fname+"/train")
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader, fname+"/val")
        test_acc = eval(args, model, device, test_loader, fname+"/test")

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        save_best_model(val_acc, test_acc, epoch, model, optimizer, criterion, fname)
        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")
        if epoch%10==0:
            torch.save(model.state_dict(), fname+"/model.pth")
    print(f'dataset : {args.dataset}')
    print(f'best model test auc : {save_best_model.best_test_loss}')
    with open(fname+'/done', 'w') as f:
        f.write(f'best model test auc : {save_best_model.best_test_loss}\n model_weight : {model_weight} ')
    torch.save(model.state_dict(), fname+"/model.pth")
    if not args.filename == "":
        writer.close()
    send_msg(f"{args.dataset} finetune completed")
if __name__ == "__main__":
    main()
