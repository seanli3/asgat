from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.metrics import f1_score
import numpy as np
# from torch_sparse import spmm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def run(dataset, model, runs, epochs, lr, weight_decay, patience,
        permute_masks=None, logger=None):

    durations = []
    for _ in range(runs):
        data = dataset[0]
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        best_val_acc = float(0)
        eval_info_early_model = None
        bad_counter = 0

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch
            # if epoch % 10 == 0:
                # print(eval_info)

            if logger is not None:
                logger(eval_info)

            if eval_info['val_acc'] > best_val_acc or eval_info['val_loss'] < best_val_loss:
                if eval_info['val_acc'] >= best_val_acc and eval_info['val_loss'] <= best_val_loss:
                    eval_info_early_model = eval_info
                    # torch.save(model.state_dict(), './best_{}_gat.pkl'.format(dataset.name))
                best_val_acc = np.max((best_val_acc, eval_info['val_acc']))
                best_val_loss = np.min((best_val_loss, eval_info['val_loss']))
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    duration = tensor(durations)

    print('Early stop! Min val loss: ', best_val_loss, ', Max val accuracy: ', best_val_acc)
    print('Early stop model validation loss: ', eval_info_early_model['val_loss'], ', accuracy: ', eval_info_early_model['val_acc'])
    print('Early stop model test accuracy: ', eval_info_early_model['test_acc'], ', f1-score: ', eval_info_early_model['f1_score'])
    print('Duration: {:.3f}'.format(duration.mean().item()))
    return eval_info_early_model['test_acc']


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)[0]
    # coefficients = torch.eye(filterbanks[0].shape[0], filterbanks[0].shape[1])
    # for c in filterbanks:
    #     coefficients = spmm(c.indices(), c.values(), c.shape[0], c.shape[1], coefficients)
    # discrimative_loss = coefficients.mean()
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits  = model(data)[0]

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    outs['f1_score'] = f1_score(data.y.cpu(), logits.max(1)[1].cpu(), average='micro')

    return outs
