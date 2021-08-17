from __future__ import division

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.metrics import f1_score
import numpy as np
# from torch_sparse import spmm


def run(use_dataset, Model, runs, epochs, lr, weight_decay, patience, logger=None, cuda=False):
    device = torch.device('cuda' if cuda else 'cpu')
    val_losses, train_accs, val_accs, test_accs, test_micro_f1s, test_macro_f1s, durations = [], [], [], [], [], [], []
    for _ in range(runs):
        dataset = use_dataset()
        data = dataset[0]
        model = Model(dataset)
        if cuda:
            torch.cuda.synchronize()

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if cuda:
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
            # if epoch % 1 == 0:
            #     print(eval_info)

            if logger is not None:
                logger(eval_info)

            if eval_info['val_acc'] > best_val_acc or eval_info['val_loss'] < best_val_loss:
                if eval_info['val_acc'] >= best_val_acc and eval_info['val_loss'] <= best_val_loss:
                    eval_info_early_model = eval_info
                    # torch.save(model.state_dict(), './best_{}_mlp.pkl'.format(dataset.name))
                best_val_acc = np.max((best_val_acc, eval_info['val_acc']))
                best_val_loss = np.min((best_val_loss, eval_info['val_loss']))
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == patience:
                    break

        if cuda:
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        val_losses.append(eval_info_early_model['val_loss'])
        train_accs.append(eval_info_early_model['train_acc'])
        val_accs.append(eval_info_early_model['val_acc'])
        test_accs.append(eval_info_early_model['test_acc'])
        test_micro_f1s.append(eval_info_early_model['test_micro_f1'])
        test_macro_f1s.append(eval_info_early_model['test_macro_f1'])
        durations.append(t_end - t_start)

    val_losses, train_accs, val_accs, test_accs, test_micro_f1s, test_macro_f1s, duration = tensor(val_losses), tensor(train_accs), tensor(val_accs), \
                                                            tensor(test_accs), tensor(test_micro_f1s), tensor(test_macro_f1s), tensor(durations)

    print('{} Val Loss: {:.4f} ± {:.3f}, Train Accuracy: {:.3f} ± {:.3f}, Val Accuracy: {:.3f} ± {:.3f}, Test Accuracy: {:.3f} ± {:.3f}, Micro-F1: {:.3f} ± {:.3f}, Macro-F1: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(dataset.name,
                 val_losses.mean().item(),
                 val_losses.std().item(),
                 train_accs.mean().item(),
                 train_accs.std().item(),
                 val_accs.mean().item(),
                 val_accs.std().item(),
                 test_accs.mean().item(),
                 test_accs.std().item(),
                 test_micro_f1s.mean().item(),
                 test_micro_f1s.std().item(),
                 test_macro_f1s.mean().item(),
                 test_macro_f1s.std().item(),
                 duration.mean().item()))


    # print('row_diff:', cal_row_diff(model, data), 'col_diff:', cal_col_diff(model, data))
    return test_accs.mean().item()


def cal_col_diff(model, data):
    with torch.no_grad():
        model.eval()
        _, embeddings = model(data)
        test_embeddings = embeddings[data.test_mask]
        normalized_test_embeddings = test_embeddings/torch.linalg.norm(test_embeddings, 1, dim=0)
        index = list(range(test_embeddings.shape[1]))
        sum = 0
        for _ in range(test_embeddings.shape[1]):
            index.insert(0, index.pop())
            sum += torch.linalg.norm(normalized_test_embeddings - normalized_test_embeddings[:, index], 2, dim=0).sum()
        col_diff = sum / pow(test_embeddings.shape[1], 2)
    return col_diff



def cal_row_diff(model, data):
    with torch.no_grad():
        model.eval()
        _, embeddings = model(data)
        test_embeddings = embeddings[data.test_mask]
        normalized_test_embeddings = test_embeddings/torch.linalg.norm(test_embeddings, 1, dim=1).view(-1, 1)
        # calculate row-diff
        index = list(range(test_embeddings.shape[0]))
        sum = 0
        for _ in range(test_embeddings.shape[0]):
            index.insert(0, index.pop())
            sum += torch.linalg.norm(normalized_test_embeddings - normalized_test_embeddings[index], 2, dim=1).sum()
        row_diff = sum / pow(test_embeddings.shape[0], 2)
    return row_diff


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
        logits = model(data)[0]

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

        outs['{}_micro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='micro')
        outs['{}_macro_f1'.format(key)] = f1_score(data.y[mask].cpu(), logits[mask].max(1)[1].cpu(), average='macro')

    return outs
