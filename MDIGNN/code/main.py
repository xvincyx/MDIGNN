import pandas as pd
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from argparse import ArgumentParser
from layer import MagNet
from utils.preprocess import geometric_dataset_sparse
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



cuda_device = 0
device = torch.device("cuda:%d" % cuda_device if torch.cuda.is_available() else "cpu")


def edge_dropout(L, dropout_rate=0.5):
    """Applies edge dropout to the given Laplacian matrices."""
    new_L = []
    for lap in L:
        mask = torch.rand(lap._nnz()) > dropout_rate
        indices = lap._indices()[:, mask]
        values = lap._values()[mask]
        new_L.append(torch.sparse.FloatTensor(indices, values, lap.size()))
    return new_L

def parse_args():
    parser = ArgumentParser(description="MagNet Conv (sparse version)")
    parser.add_argument('--log_root', type=str, default='../logs/',
                        help='the path saving model.t7 and the training process')
    parser.add_argument('--log_path', type=str, default='test',
                        help='the path saving model.t7 and the training process, the name of folder will be log/(current time)')
    parser.add_argument('--data_path', type=str, default='../dataset/',
                        help='data set folder, for default format see dataset/')
    parser.add_argument('--dataset', type=str, default='pancancer', help='data set selection')

    parser.add_argument('--epochs', type=int, default=3000, help='Number of (maximal) training epochs.')
    parser.add_argument('--q', type=float, default=0.25, help='q value for the phase matrix')
    parser.add_argument('--method_name', type=str, default='Magnet', help='method name')
    parser.add_argument('--K', type=int, default=1, help='K for cheb series')
    parser.add_argument('--layer', type=int, default=2, help='How many layers of gcn in the model, default 2 layers.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')
    parser.add_argument('--debug', '-D', action='store_true', help='debug mode')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=5e-3, help='l2 regularizer')
    parser.add_argument('--num_filter', type=int, default=64, help='num of filters')
    parser.add_argument('--randomseed', type=int, default=3407, help='if set random seed in training')
    # parser.add_argument('--randomseed', type=int, default=3589, help='if set random seed in training')
    return parser.parse_args()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




def main(args):
    if args.randomseed > 0:
        torch.manual_seed(args.randomseed)

    date_time = datetime.now().strftime('%m-%d-%H_%M_%S')
    log_path = os.path.join(args.log_root, args.log_path, args.save_name, date_time)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    X, label, train_masks, val_masks, L, gene_names = geometric_dataset_sparse(
        args.q, args.K, '../dataset/', load_only=False,
        save_pk=True, laplacian=True, gcn_appr=False)

    # Normalize label, the minimum should be 0 as class index
    label = label
    cluster_dim = np.amax(label) + 1

    # Convert dense Laplacian to sparse matrix
    L_img = []
    L_real = []
    for i in range(len(L)):
        L_img.append(sparse_mx_to_torch_sparse_tensor(L[i].imag).to(device))
        L_real.append(sparse_mx_to_torch_sparse_tensor(L[i].real).to(device))

    label = torch.from_numpy(label[np.newaxis]).to(device).long()  # Convert label to Long type
    X_img = torch.FloatTensor(X).to(device)
    X_real = torch.FloatTensor(X).to(device)
    # criterion = nn.NLLLoss()
    class_weights = [ 1.0,1.0/3.0]
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    splits = train_masks.shape[1]
    print(splits)



    results = np.zeros((splits, 2))

    all_probs_split = np.zeros((len(gene_names), len(range(splits))))
    AUROC = []
    AUPR = []

    all_val_true = []
    all_val_probs = []

    for split in range(splits):
        log_str_full = ''

        model = MagNet(
            in_c=X_real.size(-1),
            num_filter=args.num_filter,
            K=args.K,
            label_dim=cluster_dim,
            layer=args.layer,
            dropout=args.dropout
        ).to(device)

        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        best_test_acc = 0.0
        train_index = train_masks[:, split]
        val_index = val_masks[:, split]

        criterion = nn.CrossEntropyLoss()
        print("Train mask sum: ", np.sum(train_index))
        print("Validation mask sum: ", np.sum(val_index))

        best_test_err = float('inf')
        early_stopping = 0
        for epoch in range(args.epochs):
            start_time = time.time()
            count, train_loss, train_acc = 0.0, 0.0, 0.0

            L_real_dropout = edge_dropout(L_real, dropout_rate=0.2)
            L_img_dropout = edge_dropout(L_img, dropout_rate=0.2)

            count += np.sum(train_index)

            model.train()
            preds = model(X_real, X_img, L_real_dropout, L_img_dropout)  # Pass the dropout matrices here


            train_loss = criterion(preds[:, :, train_index], label[:, train_index])
            pred_label = preds.max(dim=1)[1]
            train_acc = 1.0 * ((pred_label[:, train_index] == label[:, train_index])).sum().detach().item() / count
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            outstrtrain = 'Train loss:, %.6f, acc:, %.3f,' % (train_loss.detach().item(), train_acc)


            with torch.no_grad():
                preds = model(X_real, X_img, L_real, L_img)  # Use original matrices for validation
                pred_probs = torch.softmax(preds, dim=1)
                pred_label = pred_probs.max(dim=1)[1]

                class1_probs = pred_probs[:, 1, :].cpu().numpy()
                all_probs_split[:, split] = class1_probs.flatten()

                val_probs = pred_probs[:, 1, val_index].cpu().numpy()
                val_true = label[:, val_index].cpu().numpy()
                val_true = val_true[0]

                val_probs = val_probs.ravel()

                all_val_true.extend(val_true)
                all_val_probs.extend(val_probs)

                test_loss = criterion(preds[:, :, val_index], label[:, val_index])
                val_auroc = roc_auc_score(val_true, val_probs)
                val_aupr = average_precision_score(val_true, val_probs)

                outstrval = f'Validation loss: {test_loss.item():.6f}, AUROC: {val_auroc:.6f}, AUPR: {val_aupr:.6f}'

            duration = "---, %.4f, seconds ---" % (time.time() - start_time)
            log_str = ("%d ,/, %d ,epoch," % (epoch, args.epochs)) + outstrtrain + outstrval + duration
            log_str_full += log_str + '\n'

            ####################
            # Save weights
            ####################
            save_perform = test_loss.detach().item()
            if save_perform <= best_test_err:
                early_stopping = 0
                best_test_err = save_perform
                torch.save(model.state_dict(), log_path + '/model' + str(split) + '.t7')
            else:
                early_stopping += 1
            # if early_stopping > 500 or epoch == (args.epochs - 1):
            #     torch.save(model.state_dict(), log_path + '/model_latest' + str(split) + '.t7')
            #     break


            print(
                f"Epoch {epoch}, Save Perform: {save_perform}, Best Test Err: {best_test_err}, Early Stopping: {early_stopping}")

            if early_stopping > 100 or epoch == (args.epochs - 1):
                torch.save(model.state_dict(), log_path + '/model_latest' + str(split) + '.t7')
                print(f"Early stopping triggered at epoch {epoch}. Model saved as model_latest{str(split)}.t7")
                break


        logstr = f'AUROC (val): {val_auroc:.6f}, AUPR (val): {val_aupr:.6f}'
        print(logstr)
        AUROC.append(val_auroc)
        AUPR.append(val_aupr)

        results[split] = [val_auroc, val_aupr]
        log_str_full += logstr
        with open(log_path + '/log' + str(split) + '.csv', 'w') as file:
            file.write(log_str_full)
            file.write('\n')
        torch.cuda.empty_cache()

    mean_auroc = sum(AUROC) / len(AUROC)
    mean_aupr = sum(AUPR) / len(AUPR)
    logstr = f'5-FOLD-AUROC (val): {mean_auroc:.6f}, 5-FOLD-AUPR (val): {mean_aupr:.6f}'
    print(logstr)


    combined_mask = train_masks | val_masks
    known_label_indices = np.where(combined_mask.sum(axis=1) > 0)[0]


    mean_probs = np.mean(all_probs_split, axis=1)

    results_df = pd.DataFrame({
        'Gene Name': gene_names,
        'Mean Probability (Class 1)': mean_probs
    })



    #####################################################################
    torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    args = parse_args()
    if args.debug:
        args.epochs = 1

    dir_name = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../result_arrays', args.log_path + '/')
    args.log_path = os.path.join(args.log_path, args.method_name)

    if os.path.isdir(dir_name) == False:
        try:
            os.makedirs(dir_name)
        except FileExistsError:
            print('Folder exists!')
    save_name = args.method_name + 'lr' + str(int(args.lr * 1000)) + 'num_filters' + str(
        int(args.num_filter)) + 'q' + str(int(100 * args.q)) + 'layer' + str(int(args.layer)) + 'K' + str(int(args.K))
    args.save_name = save_name
    results = main(args)
    np.save(dir_name + save_name, results)
