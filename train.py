import os
import argparse
from pathlib import Path
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from metrics import get_chains_to_merge, get_interface, get_raw_mapping, get_values_from_lddt_predictions, get_values_from_lddt_results, parse_chains
from network.resEGNN import resEGNN_with_ne
from pdb_utils_crank import merge_chains
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to train feature dataset.')
    parser.add_argument('--set-train', type=str, required=False, default='train.txt',
                        help='Path to txt file with train sample names')
    parser.add_argument('--validation', type=str, required=True,
                        help='Path to validation feature dataset.')
    parser.add_argument('--set-validation', type=str, required=True, default='valid.txt',
                        help='Path to txt file with validation sample names')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save model weights.')
    parser.add_argument('--cpu', action='store_true', default=False, help='Force to use CPU.')
    parser.add_argument('--epochs', type=int, required=False, default=60)

    parser.add_argument('--w_dist', type=float, required=False, default=1.0, help='distance loss weight')
    parser.add_argument('--w_bin', type=float, required=False, default=1.0, help='bin loss weight')
    parser.add_argument('--w_score', type=float, required=False, default=5.0, help='score loss weight')

    parser.add_argument('--lr', type=float, required=False, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00005, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu else 'cpu'
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    path_docked=Path('/mnt/volume_complex_lddt/consistent')
    # /mnt/volume_complex_lddt/consistent_alpha_hedge
    dim2d = 25 + 9 * 5
    model = resEGNN_with_ne(dim2d=dim2d, dim1d=33)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    with open(args.set_train) as f:
        train_sample=f.read().splitlines()
    with open(args.set_validation) as f:
        val_sample=f.read().splitlines()
    count_val_loss_decr=0
    writer = SummaryWriter()
    for i in range(args.epochs):
        if count_val_loss_decr>10:
            break
        loss_score_train=0
        loss_bin_train=0
        loss_dist_train=0
        train_loss_sum = 0
        total_size_train = 0
        model.train()
        for sample in tqdm(train_sample):
            path_sample=args.train + '/' + sample+'.pt'
            if not os.path.exists(path_sample):
                warnings.warn(f'File with fatures for {sample} does not exists in {args.train}')
                continue
            x = torch.load(args.train + '/' + sample+'.pt')
            f1d = x['f1d'].to(device)
            f2d = x['f2d'].to(device)
            pos = x['pos'].to(device)
            el = [i.to(device) for i in x['el']]

            label_lddt = x['label_lddt'].to(device)
            diff_bins = x['diff_bins'].to(device)
            pos_transformed = x['pos_transformed'].to(device)

            pred_bin, pred_pos, pred_lddt = model(f1d, f2d, pos, el)
    
            path_to_complex=path_docked / sample
            ab_chains,ag_chains=parse_chains(sample)
            ab_chains,ag_chains=get_chains_to_merge(path_to_complex  / 'real.pdb',ab_chains)
            merge_chains(ab_chains,ag_chains,'ref_temp.pdb')
            mapping=get_raw_mapping(path_to_complex,joined_path=path_to_complex/'real_joined.pdb')
            contacts=get_interface(Path('ref_temp.pdb'))
            label_lddt_interface=get_values_from_lddt_results(path_to_complex, mapping, contacts)
            pred_lddt_interface=get_values_from_lddt_predictions(path_to_complex, mapping, contacts,pred_lddt)
            loss_score = F.smooth_l1_loss(pred_lddt_interface, torch.tensor(label_lddt_interface,device=device))
            loss_bin = F.cross_entropy(pred_bin, diff_bins)
            loss_dist = F.mse_loss(torch.nn.functional.pdist(pred_pos),
                                   torch.nn.functional.pdist(pos_transformed))
            total_loss = args.w_dist * loss_dist + args.w_bin * loss_bin + args.w_score * loss_score
            loss_score_train+=loss_score
            loss_bin_train+=loss_bin
            loss_dist_train+=loss_dist
            train_loss_sum += total_loss.detach().cpu().tolist()
            total_size_train += 1
            if total_size_train % args.batch_size == 0:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        print("Epoch: {} Train loss: {:.4f}".format(i, train_loss_sum / total_size_train))
        loss_score_valid=0
        loss_bin_valid=0
        loss_dist_valid=0
        val_loss_sum = 0
        total_size_valid = 0
        model.eval()
        for sample in val_sample:
            path_sample_val=args.validation + '/' + sample+'.pt'
            if not os.path.exists(path_sample_val):
                warnings.warn(f'File with fatures for {sample} does not exists in {args.validation}')
                continue
            x = torch.load(args.validation + '/' + sample+'.pt')
            f1d = x['f1d'].to(device)
            f2d = x['f2d'].to(device)
            pos = x['pos'].to(device)
            el = [i.to(device) for i in x['el']]
        

            label_lddt = x['label_lddt'].to(device)
            diff_bins = x['diff_bins'].to(device)
            pos_transformed = x['pos_transformed'].to(device)
            with torch.no_grad():
                pred_bin, pred_pos, pred_lddt = model(f1d, f2d, pos, el)
            path_to_complex=path_docked / sample
            ab_chains,ag_chains=parse_chains(sample)
            ab_chains,ag_chains=get_chains_to_merge(path_to_complex  / 'real.pdb',ab_chains)
            merge_chains(ab_chains,ag_chains,'ref_temp.pdb')
            mapping=get_raw_mapping(path_to_complex,joined_path=path_to_complex/'real_joined.pdb')
            contacts=get_interface(Path('ref_temp.pdb'))
            label_lddt_interface=get_values_from_lddt_results(path_to_complex, mapping, contacts)
            pred_lddt_interface=get_values_from_lddt_predictions(path_to_complex, mapping, contacts,pred_lddt)

            loss_score = F.smooth_l1_loss(pred_lddt_interface, torch.tensor(label_lddt_interface,device=device))
            loss_bin = F.cross_entropy(pred_bin, diff_bins)
            loss_dist = F.mse_loss(torch.nn.functional.pdist(pred_pos),
                                   torch.nn.functional.pdist(pos_transformed))
            total_loss = args.w_dist * loss_dist + args.w_bin * loss_bin + args.w_score * loss_score
            loss_score_valid+=loss_score
            loss_bin_valid+=loss_bin
            loss_dist_valid+=loss_dist

            val_loss_sum += total_loss.detach().cpu()
            total_size_valid += 1
        if not i:
            val_loss_prev=val_loss_sum
        else:
            if val_loss_sum<val_loss_prev:
                count_val_loss_decr=0
            else:
                count_val_loss_decr+=1
            val_loss_prev=val_loss_sum
        print("Epoch: {} Validation loss: {}".format(i, val_loss_sum / total_size_valid))
        writer.add_scalars('loss_score',{'train':loss_score_train/total_size_train,'validation':loss_score_valid/total_size_valid},i)
        writer.add_scalars('loss_bin',{'train':loss_bin_train/total_size_train,'validation':loss_bin_valid/total_size_valid},i)
        writer.add_scalars('loss_dist',{'train':loss_dist_train/total_size_train,'validation':loss_dist_valid/total_size_valid},i)
        writer.add_scalars('total_loss',{'train':train_loss_sum / total_size_train,'validation':val_loss_sum/total_size_valid},i)
        writer.close()
    torch.save(model.state_dict(), os.path.join(args.output, 'model_weights.pth'))

# python3 train.py --train outputs/processed/ --validation outputs/processed/ --output outputs/ --epochs 15
# python3 train.py --train /mnt/volume/outputs_vacation/processed --validation /mnt/volume/outputs_vacation/processed --output outputs/ --epochs
# python3 train.py --train /mnt/volume/outputs_vacation/processed --validation /mnt/volume/outputs_vacation/processed --output outputs/ --epochs 2 --set-train train_small.txt --set-validation valid_small.txt
# python3 train.py --train /mnt/volume/outputs_vacation/processed --validation /mnt/volume/outputs_vacation/processed --output outputs/ --epochs 2 --set-train train.txt --set-validation valid.txt 
