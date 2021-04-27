import os
import argparse
import logging
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--dataset', choices=['PIE', 'FR'], default='PIE')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--name', default='demo', help='output dir')

    return parser.parse_args()


def get_root_logger(args):
    if not os.path.exists(f'out/{args.name}'):
        os.mkdir(f'out/{args.name}')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f'out/{args.name}/log.txt', mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def evaluate(args, model, loader):
    def pred(x):
        _, id = torch.max(x, dim=1)
        return id.cpu().numpy()

    model.eval()
    true_ys = np.array([], dtype=np.int)
    pred_ys = np.array([], dtype=np.int)
    for x, y in loader:
        if args.cuda:
            x = x.cuda()
            y = y.cuda()

        y_hat = model(x)
        true_ys = np.concatenate((true_ys, y.squeeze(-1).cpu().numpy()))
        pred_ys = np.concatenate((pred_ys, pred(y_hat)))

    return precision_score(true_ys, pred_ys, average='macro'), \
           recall_score(true_ys, pred_ys, average='macro'), \
           f1_score(true_ys, pred_ys, average='macro')
