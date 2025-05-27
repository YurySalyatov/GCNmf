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
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

torch.set_warn_always(False)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

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
parser.add_argument('--epoch', default=10000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=100, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')

args = parser.parse_args()
dataset_str = args.dataset
noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
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


def feature_noise(data, percentage):
    noisy_data = data.clone()
    noisy_data.to(device)
    tensor = noisy_data.features
    if percentage <= 0:
        return noisy_data

    num_features = tensor.size(1)
    # print("num_features", num_features)
    num_selected_features = int(percentage * num_features)
    # print("num_selected_features", num_selected_features)
    if num_selected_features == 0:
        return noisy_data

    # Выбираем случайные фичи
    selected_features = torch.randperm(num_features, device=device)[:num_selected_features]
    # print("selected_features", selected_features)
    # Генерируем значения для замены
    flattened = tensor.flatten()
    shuffled_values = flattened[torch.randperm(len(flattened), device=device)][:tensor.size(0) * num_selected_features]
    replacement = shuffled_values.view(tensor.size(0), num_selected_features)

    # Создаем копию и применяем шум
    noised_tensor = tensor.clone()
    # print("noisy tensor")
    # print(noised_tensor[:, selected_features])
    noised_tensor[:, selected_features] = replacement
    # noised_tensor[:, selected_features] = torch.rand_like(noised_tensor[:, selected_features])
    # print(noised_tensor[:, selected_features])
    noisy_data.features = noised_tensor
    return noisy_data


if __name__ == '__main__':
    methods = [feature_noise]
    for dataset_name in ['cora', 'citeseer']:
        print(dataset_name)
        data = NodeClsData(dataset_name)
        for method in methods:
            tables_dir = f"results/{dataset_name}/{method.__name__}"
            os.makedirs(tables_dir, exist_ok=True)
            results = []
            print(method.__name__)
            for sigma in noise_levels:
                one_result = {"sigma": sigma}
                pu_arr = []
                acc_arr = []
                for _ in range(5):
                    noisy_data = method(data, sigma)
                    noisy_data.to('cpu')
                    model = GCNmf(noisy_data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
                    params = {
                        'lr': args.lr,
                        'weight_decay': args.wd,
                        'epochs': args.epoch,
                        'patience': args.patience,
                        'early_stopping': True
                    }
                    trainer = NodeClsTrainer(noisy_data, model, params, niter=20, verbose=args.verbose)
                    max_acc = trainer.run()['max_acc']
                    model.load_state_dict(torch.load("trained_model/without_noisy.pkl"))
                    model.to(device)
                    noisy_data.to(device)

                    # Оценка PU
                    num_samples = 20
                    predictions = []
                    model.train()
                    for _ in range(num_samples):
                        with torch.no_grad():
                            log_probs = model(noisy_data)
                            predictions.append(torch.exp(log_probs[data.test_mask]))

                    predictions = torch.stack(predictions)
                    mean_pred = predictions.mean(dim=0)
                    mean_pred_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-18), dim=1)
                    pu = mean_pred_entropy.mean()
                    pu_arr.append(pu)
                    acc_arr.append(max_acc)
                pu_arr = torch.stack(pu_arr)
                acc_arr = torch.stack(acc_arr)
                mean_pu = pu_arr.mean().item()
                var_pu = pu_arr.var().item()
                one_result[f" PU"] = mean_pu
                one_result[f" var PU"] = var_pu
                one_result[f" max acc"] = acc_arr.max().item()
                one_result[f" min acc"] = acc_arr.min().item()
                one_result[f" mean acc"] = acc_arr.mean().item()
                one_result[f" var acc"] = acc_arr.var().item()
                print(one_result)
                results.append(one_result)
            # plot_dir = f"results/{dataset_name}/{method.__name__}/plots"
            # os.makedirs(plot_dir, exist_ok=True)
            table_file = f"results/{dataset_name}/{method.__name__}/table_experiment2.txt"
            os.makedirs(f"results/{dataset_name}/{method.__name__}", exist_ok=True)

            # plot_all_results(results, save_path=plot_dir)
            save_table(results, filename=table_file)
