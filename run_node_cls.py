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
noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    def add_noisy_features(data, noise_level, convert=True):
          noisy_data = data.clone()
          if convert:
              noisy_data.to(device)
          a, b = data.features.shape
          if convert:
            # Конвертируем константы в тензоры на нужном устройстве
              feature_mean_tensor = torch.tensor(feature_mean[dataset_str], 
                                                dtype=torch.float32, 
                                                device=device)  # (2)
              feature_std_tensor = torch.tensor(feature_std[dataset_str], 
                                              dtype=torch.float32, 
                                              device=device)    # (2)
          else:
              feature_mean_tensor = torch.tensor(feature_mean[dataset_str], 
                                                dtype=torch.float32)  # (2)
              feature_std_tensor = torch.tensor(feature_std[dataset_str], 
                                              dtype=torch.float32)    # (2)
          
          # Создаем шум НА GPU (если device='cuda')
          if convert:
              noise = torch.randn(a, b, device=device)  # (3)
          else:
              noise = torch.randn(a, b)
          
          
          # Вычисляем шумовую компоненту
          noise_component = (feature_mean_tensor + 2 * noise * torch.sqrt(feature_std_tensor)) * noise_level
          
          # Применяем к features (уже на GPU)
          noisy_data.features += noise_component
          
          return noisy_data
    for main_sigma in noise_levels:
        data = NodeClsData(args.dataset)
        data = add_noisy_features(data, main_sigma, False)
        mask = generate_mask(data.features, args.rate, args.type)
        apply_mask(data.features, mask)
        model = GCNmf(data, nhid=args.nhid, dropout=args.dropout, n_components=args.ncomp)
        params = {
            'lr': args.lr,
            'weight_decay': args.wd,
            'epochs': args.epoch,
            'patience': args.patience,
            'early_stopping': True
        }
        trainer = NodeClsTrainer(data, model, params, niter=10, verbose=args.verbose)
        trainer.run()
        model.load_state_dict(torch.load("trained_model/without_noisy.pkl"))
        fix_model = model
        # fix_model.to(device)
        noisy_data = data.clone()
        noisy_data.to(device)

        def compute_entropy(log_probs):
            return -torch.sum(torch.exp(log_probs) * log_probs, dim=1)


        def compute_margin(log_probs):
            exp = torch.exp(log_probs)
            exp, _ = torch.sort(exp, dim=1, descending=True)
            return exp[:, 0] - exp[:, 1]


        results = []
        pu_model = []
        for sigma in noise_levels[1:]:
            # Добавление шума
            noisy_data = add_noisy_features(data, sigma)
            # print("shape noisy data", noisy_data.x.shape)
            # print(fix_model)
            # Оценка PU
            fix_model.eval()
            with torch.no_grad():
                log_probs = fix_model(noisy_data)
                # print("out", out.shape)
                # print(probs.shape)
                entropy = compute_entropy(log_probs[noisy_data.test_mask])
                margin = compute_margin(log_probs[noisy_data.test_mask])
                # print(entropy.shape)
                e_pu = entropy.mean().item()
                m_pu = margin.mean().item()

            # Оценка AU (100 итераций с шумом)
            num_samples = 100
            predictions = []
            perturbed_data = data.clone()
            perturbed_data.to(device)
            fix_model.train()
            fix_model.dropout = sigma
            entropy = []
            for _ in range(num_samples):
                with torch.no_grad():
                    log_probs = fix_model(perturbed_data)[perturbed_data.test_mask]
                    predictions.append(log_probs)
                    entropy.append(compute_entropy(log_probs))

            predictions = torch.stack(predictions)
            entropy = torch.stack(entropy)
            # print(predictions.shape)
            mean_pred = predictions.mean(dim=0)
            mean_pred_entropy = compute_entropy(mean_pred)
            mean_entropy_pred = entropy.mean(dim=0)
            # au_arr = torch.var(predictions, dim=0)
            au = mean_entropy_pred.mean().item()
            # print(au_arr.shape)
            # au = au_arr.mean().item()
            results.append({'sigma': sigma, 'PU': e_pu, 'M PU': m_pu, 'AU': au})
            print(f"Noise: {sigma:.2f} | Entropy PU: {e_pu:.4f} | Margin PU: {m_pu:.4f} | AU: {au:.4f}")


        # Визуализация
        def plt_res(name):
            plt.figure(figsize=(10, 6))
            plt.plot(
                [sigma for sigma in noise_levels][1:],
                [res[name] for res in results],
                marker='o',
                label=f'{name} (Entropy)'
            )
            plt.xlabel('Noise Level')
            plt.ylabel('Uncertainty')
            plt.legend()
            plt.title('Dependence of P Uncertainties on Noise Level')
            plt.grid(True)
            plt.show()


        # print(f"A Uncertainties:", au)
        # us = ['PU', 'AU', 'M PU']
        # for u in us:
        #     plt_res(u)

        data = list(zip(noise_levels[1:], [res['PU'] for res in results], [res['AU'] for res in results],
                        [res['M PU'] for res in results]))
        headers = ["Noisy Level", "PU", "AU", "M PU"]
        table_str = tabulate(
            data, 
            headers=headers, 
            tablefmt="grid", 
            floatfmt=".5f"
        )

        # Сохраняем в файл
        with open(f"results2_GCNmf_{main_sigma}.txt", "w", encoding="utf-8") as f:
            f.write(table_str)
        # print(tabulate(data,
        #                headers=headers,
        #                tablefmt="grid",  # можно изменить на "simple", "fancy_grid" и др.
        #                floatfmt=".5f"))  # формат чисел
