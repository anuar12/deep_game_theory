import numpy as np
import torch
from torch.autograd import Variable


def nll(pred, actual):
    return -torch.sum(actual * torch.log(pred))


def eval_data(args, data_dict, model):
    ''' Evaluate a dictionary of games '''
    if type(args) != dict:
        args = vars(args)
    normalize_constant = 1000.
    nll_tot = 0.
    n = 0
    n_game_shapes = len(data_dict)
    accuracies_by_sample = []
    for k, v in data_dict.iteritems():
        X, y = prep(v, k)
        X /= normalize_constant
        n += y.sum()
        X = Variable(torch.from_numpy(X), requires_grad=False)
        y = Variable(torch.from_numpy(y), requires_grad=False)
        if args['cuda']:
            X = X.cuda()
            y = y.cuda()
        nll_tot += nll(model(X), y.clone())
        accuracies_by_sample += accuracy(model(X), y.clone()).tolist()
    acc_average = float(sum(accuracies_by_sample)) / len(accuracies_by_sample)
    return nll_tot, n, acc_average


def accuracy(pred, actual):
    pred = pred.cpu().data.numpy()
    pred[pred < (pred.max(axis=1) - 1e-10).reshape(-1, 1)] = 0.
    acc = np.logical_and(pred, actual.cpu().data.numpy())
    acc_by_sample = acc.sum(axis=1)
    return acc_by_sample


def data_entropy(data_dict):
    ''' Should be non-zero when more than one action is in the labels '''
    entropy = 0.
    for k, v in data_dict.iteritems():
        X, y = prep(v, k)
        probs = y / y.sum(axis=1).reshape(-1, 1)
        entropy += np.sum(y * np.log(probs + 1e-8))
    return -entropy


def prep(data, key):
    '''
    Reshape games
    '''
    X, Y = data
    n, _ = X.shape
    X = X.reshape((n, 2, key[0], key[1]))
    X = np.array(X, dtype="float32")
    Y = np.array(Y, dtype="float32")
    return X, Y


def apply_l1(model, loss, l1_val):
    reg_loss = 0.
    for name, param in model.named_parameters():
        if 'mixture' in name or 'bias' in name or 'sharp' in name or\
                'threshold' in name or 'compare' in name:
            continue
        reg_loss += torch.sum(torch.abs(param))
    loss += l1_val * reg_loss
    return loss


def uniform_loss(data):
    total_loss = 0.
    for game_shape in data:
        y = data[game_shape][1]
        preds = np.ones((y.shape[0], y.shape[1])) / y.shape[1]
        total_loss += np.sum(np.multiply(np.log(preds), y))
    return -total_loss


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it //wtains layers let call it recursively to get params and weights
        if type(module) in [
                torch.nn.modules.container.Container,
                torch.nn.modules.container.Sequential]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = torch.nn.modules.module._addindent(modstr, 2)
        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   
    tmpstr = tmpstr + ')'
    return tmpstr


