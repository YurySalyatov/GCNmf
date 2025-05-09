"""
version 1.0
date 2021/02/04
"""

import argparse
import torch
from models import GCNmf
from train import NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import os

feature_mean = {'citeseer': 0.0002688338, 'cora': 0.0023735422}
feature_std = {'citeseer': 0.0006978367, 'cora': 0.0056594303}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--nhid', default=16, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=500, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')

args = parser.parse_args()
dataset_str = args.dataset
noise_levels = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_table(results, filename="results_experiment1_table.txt"):
    all_keys = [k for k in results[0].keys() if k != 'sigma']
    # Создаем список форматов: первый столбец .2f, остальные .5f
    headers = ["Noise Level"] + all_keys
    float_fmts = [".2f"] + [".5f"] * (len(headers) - 1)
    data = [[res[key] for res in results] for key in results[0].keys()]
    rows = list(zip(*data))
    # Генерируем таблицу
    table_str = tabulate(
        rows,
        headers=headers,
        tablefmt="grid",
        floatfmt=float_fmts
    )

    # Сохраняем в файл
    with open(filename, "w") as f:
        f.write(table_str)

    return table_str


def node_noise(data, percentage, convert=True):
    """
    Заменяет все фичи у случайно выбранного процента вершин на значения из общего распределения тензора
    Args:
        tensor: исходный тензор (num_nodes, num_features)
        percentage: процент вершин для замены (0.0 - 1.0)
    Returns:
        тензор с шумом
    """
    tensor = data.features
    if percentage <= 0:
        res = data.clone()
        if convert:
            res.to(device)
        return res

    num_nodes = tensor.size(0)
    num_selected = int(percentage * num_nodes)

    if num_selected == 0:
        res = data.clone()
        if convert:
            res.to(device)
        return res

    # Выбираем случайные вершины
    selected_nodes = torch.randperm(num_nodes)[:num_selected]

    # Генерируем значения для замены из общего распределения
    flattened = tensor.flatten()
    shuffled_values = flattened[torch.randperm(len(flattened))][:num_selected * tensor.size(1)]
    replacement = shuffled_values.view(num_selected, tensor.size(1))

    # Создаем копию и применяем шум
    noised_tensor = tensor.clone()
    noised_tensor[selected_nodes] = replacement
    noisy_data = data.clone()
    noisy_data.features = noised_tensor
    if convert:
        noisy_data.to(device)
    return noisy_data


def feature_noise(data, percentage, convert=True):
    """
    Заменяет случайный процент фичей для всех вершин на значения из общего распределения тензора
    Args:
        tensor: исходный тензор (num_nodes, num_features)
        percentage: процент фичей для замены (0.0 - 1.0)
    Returns:
        тензор с шумом
    """
    tensor = data.features
    if percentage <= 0:
        res = data.clone()
        if convert:
            res.to(device)
        return res

    num_features = tensor.size(1)
    num_selected_features = int(percentage * num_features)

    if num_selected_features == 0:
        res = data.clone()
        if convert:
            res.to(device)
        return res

    # Выбираем случайные фичи
    selected_features = torch.randperm(num_features)[:num_selected_features]

    # Генерируем значения для замены
    flattened = tensor.flatten()
    shuffled_values = flattened[torch.randperm(len(flattened))][:tensor.size(0) * num_selected_features]
    replacement = shuffled_values.view(tensor.size(0), num_selected_features)

    # Создаем копию и применяем шум
    noised_tensor = tensor.clone()
    noised_tensor[:, selected_features] = replacement
    noisy_data = data.clone()
    noisy_data.features = noised_tensor
    if convert:
        noisy_data.to(device)
    return noisy_data


if __name__ == '__main__':
    methods = [feature_noise, node_noise]
    for dataset_name in ['cora', 'citeseer']:
        data = NodeClsData(dataset_name)
        for method in methods:
            tables_dir = f"results/{dataset_name}/{method.__name__}"
            os.makedirs(tables_dir, exist_ok=True)
            results = []
            print(method.__name__)
            for sigma in noise_levels:
                noisy_data = method(data, sigma, False)
                mask = generate_mask(noisy_data.features, args.rate, args.type)
                apply_mask(noisy_data.features, mask)
                model = GCNmf(noisy_data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
                params = {
                    'lr': args.lr,
                    'weight_decay': args.wd,
                    'epochs': args.epoch,
                    'patience': args.patience,
                    'early_stopping': True
                }
                trainer = NodeClsTrainer(noisy_data, model, params, niter=10, verbose=args.verbose)
                trainer.run()
                model.load_state_dict(torch.load("trained_model/without_noisy.pkl"))
                model.to(device)
                noisy_data.to(device)


                def compute_entropy(log_probs):
                    return -torch.sum(torch.exp(log_probs) * log_probs, dim=1)


                # Оценка AU (100 итераций с шумом)
                num_samples = 100
                model.train()
                model.set_train()
                entropy = []
                predictions = []
                for _ in range(num_samples):
                    with torch.no_grad():
                        log_probs = model(noisy_data)
                        predictions.append(torch.exp(log_probs))
                        entropy.append(compute_entropy(log_probs))

                predictions = torch.stack(predictions)
                entropy = torch.stack(entropy)
                mean_pred = predictions.mean(dim=0)
                mean_pred_entropy = -torch.sum(mean_pred * torch.log(mean_pred), dim=1)
                mean_entropy_pred = entropy.mean(dim=0)
                d_au = mean_entropy_pred.mean().item()
                d_pu = mean_pred_entropy.mean().item()
                mu = d_pu - d_au
                results.append(
                    {'sigma': sigma, "Dropout PU": d_pu, 'Dropout AU': d_au, "Dropout MU": mu})
                print(f"noisy: {sigma}")
                print(f"Dropout PU: {d_pu:.4f}, Dropout AU: {d_au:.4f}, Dropout MU: {mu:.4f}")
                save_table(results, tables_dir + f"/GCNmf_experiment2.txt")
