from __future__ import print_function, absolute_import
import time
import os
import math
import pprint as pp
import csv
import argparse

import numpy as np
import torch
from torch.nn.init import xavier_uniform

import utils as u
import deepch.data
from deepch.models import DeepCH
from deepch.layers import FeatureLayers
from attention.attend import AttentionNet
from attention.synthetic_data import gen_many_rand_games, gen_all_games,\
                        gen_many_dom_games, gen_many_iter_dom_games


''' Sample call:
    python train.py --attention True --synthetic_data all --min_size 3 --max_size 3 --lr 5e-4 --att_hid_layers 1 --att_hid_units 2 --epochs 2000 --batch_size 64 --plot 
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=101, type=int, help="Experiment seed")
    parser.add_argument("--fold", default=0, type=int, help="Experiment fold")
    parser.add_argument("--epochs", default=10000, type=int, help=\
                        "Number of epochs of training")
    parser.add_argument("--batch_size", default=0, type=int, help=\
                        "Number of game shapes in a batch for gradient update")
    parser.add_argument("--lr", default=4e-4, type=float, help="Learning rate")
    parser.add_argument("--l1", default=0.01, type=float, help="L1 Regularization")
    parser.add_argument("--att_hid_units", default=2, type=int,
                        help="Number of Hidden Units in Attention layers")
    parser.add_argument("--att_hid_layers", default=2, type=int,
                        help="Number of Attention layers")
    parser.add_argument("--original", default=False, type=bool,
                        help="Orignal NIPS model")
    parser.add_argument("--attention", default=True, type=bool,
                        help="Just Attention layers")
    parser.add_argument("--attention_original", default=False, type=bool,
                        help="Attention first then Original")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--synthetic_data', default='', type=str,
                        choices=['simple', 'iter_simple', 'iter_hard', 'all', ''],
                        help='Run on Synthetic Dataset')
    parser.add_argument('--min_size', default=2, type=int,
                        help='Minimum size of the synthetic game')
    parser.add_argument('--max_size', default=2, type=int,
                        help='Maximum size of the synthetic game')
    parser.add_argument('--is_more_ration_actions', action='store_true',
                        help='Make rationalizable set of cardinality more than 1')
    parser.add_argument('--plot', action='store_true', help='Plot attention')
    parser.add_argument('--dir_to_save', type=str, default='temp/',
                        help='Directory to save stdout and results')
    args = parser.parse_args()
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    if args.plot: from plot import save_att_plot, save_payoff_plot, save_att_out_plot,\
                                   prep_for_plot
    assert(args.attention != args.original)
    assert(args.attention != args.attention_original)
    return args


args = parse_args()
if args.plot: from plot import save_att_plot, save_payoff_plot, save_att_out_plot,\
                               prep_for_plot
num_train_games = 5000
print(args.synthetic_data)
if args.synthetic_data == 'simple':
    print("Dataset where only one action is dominated by all other actions for pl1!")
#    train = gen_many_rand_games(50, 5, 6)
#    test = gen_many_rand_games(50, 5, 6)
    train = gen_many_dom_games(all_games={}, num_games=50, min_size=args.min_size,
                               max_size=args.max_size, is_second_pl=False,
                               is_weak_dom=False, is_hard_labels=True)
    val = gen_many_dom_games(all_games={}, num_games=50, min_size=args.min_size,
                               max_size=args.max_size, is_second_pl=False,
                               is_weak_dom=False, is_hard_labels=True)
    test = gen_many_dom_games(all_games={}, num_games=100, min_size=args.min_size,
                               max_size=args.max_size, is_second_pl=False,
                               is_weak_dom=False, is_hard_labels=True)
elif args.synthetic_data == 'all':
    train = gen_all_games(num_games=num_train_games, min_size=args.min_size,
                          max_size=args.max_size,
                          is_more_ration_actions=args.is_more_ration_actions)
    val = gen_all_games(num_games=500, min_size=args.min_size,
                        max_size=args.max_size,
                        is_more_ration_actions=args.is_more_ration_actions)
    test = gen_all_games(num_games=500, min_size=args.min_size,
                         max_size=args.max_size,
                         is_more_ration_actions=args.is_more_ration_actions)
elif 'iter' in args.synthetic_data:
    if 'simple' in args.synthetic_data:
        print("Simple Iterated Elimination Dataset!")
        is_dec = False
    if 'hard' in args.synthetic_data:
        print("Hard Iterated Elimination Dataset!")
        is_dec = True
    
    train = gen_many_iter_dom_games(all_games={}, num_games=num_train_games,
                                    min_size=args.min_size,
                                    max_size=args.max_size,
                                    is_more_ration_actions=args.is_more_ration_actions,
                                    is_dec=is_dec)
    val = gen_many_iter_dom_games(all_games={}, num_games=500,
                                  min_size=args.min_size,
                                  max_size=args.max_size,
                                  is_more_ration_actions=args.is_more_ration_actions,
                                  is_dec=is_dec)
    test = gen_many_iter_dom_games(all_games={}, num_games=500,
                                   min_size=args.min_size,
                                   max_size=args.max_size,
                                   is_more_ration_actions=args.is_more_ration_actions,
                                   is_dec=is_dec)
else:
    dat = deepch.data.GameData("./deepch/all9.csv", normalize=50.)
    train, test = dat.train_test(args.fold, seed=args.seed)
    train = train.datalist()
    val = test.datalist()
    test = test.datalist()


if args.original:
    fl = FeatureLayers(2, [50,50], "max", dropout=0.2)
    model = DeepCH(None, fl, 50)
    model_name = 'original'
elif args.attention:
    model = AttentionNet(hid_layers=args.att_hid_layers, hid_units=args.att_hid_units,\
                         is_simult=False, is_fc_first=False, is_fc_hid=False,\
                         with_last=True, is_cuda=args.cuda, drop_p=0.) 
    model_name = 'attention'
elif args.attention_original:
    att_feats = AttentionNet(hid_layers=args.att_hid_layers, hid_units=args.att_hid_units,\
                         is_simult=False, is_fc_first=False, is_fc_hid=False,\
                         with_last=True, is_cuda=args.cuda, drop_p=0.) 
    fl = FeatureLayers(args.att_hid_units, [50,50], "max", dropout=0.2)
    model = DeepCH(att_feats, fl, 50)
    model_name = 'attention_original'
else:
    print("=== Choose which model! ===")
    raise KeyboardInterrupt


for name, module in model.named_parameters():
    if 'weight' in name:
        xavier_uniform(module)
    if 'bias' in name:
        module.data[:] = 0.

def print_params():
    for name, module in model.named_parameters():
        print(name + ':\n', module.data)


print("Using Pytorch V%s" % torch.__version__)    
print("\nArguments:")
pp.pprint(vars(args))
if args.cuda:
    print("====== Using CUDA!!! ======")
    model = model.cuda()
try:
    print("\nSample game:")
    print(train[(args.min_size, args.max_size)][0][0].reshape(2, args.min_size, args.max_size))
    print(train[(args.min_size, args.max_size)][1][0])
except:
    pass
print("\n", u.torch_summarize(model))
print("Uniform loss on Train: %.2f" % u.uniform_loss(train))
print("Uniform loss on Val: %.2f" % u.uniform_loss(val))
print("Uniform loss on Test: %.2f" % u.uniform_loss(test))
print("Data entropy on Train: %.2f" % u.data_entropy(train))
print("Data entropy on Val: %.2f" % u.data_entropy(val))
print("Data entropy on Test: %.2f" % u.data_entropy(test))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best_val_loss = np.inf
for epoch in range(args.epochs):
    optimizer.zero_grad()
    start_time = time.time()
    model.train()
    if args.batch_size > 0:
        train_games = train.items()
        ratio_to_sample_for_each_shape = float(args.batch_size) / num_train_games 
        for i in range(int(num_train_games / args.batch_size)):
            optimizer.zero_grad()
            batch_loss = 0.
            n_tot = 0
            for game_of_shape in train_games:
                game_shape, game = game_of_shape
                payoffs = game[0]
                action_counts = game[1]
                num_to_sample = math.ceil(ratio_to_sample_for_each_shape * payoffs.shape[0])
                inds_to_sample = np.random.choice(payoffs.shape[0], int(num_to_sample),\
                                                  replace=False)
                payoffs = payoffs[inds_to_sample, :]
                action_counts = action_counts[inds_to_sample, :]
                loss, n, acc = u.eval_data(args, {game_shape: [payoffs, action_counts]},\
                                                        model)
                batch_loss += loss
                n_tot += n
            batch_loss /= n_tot
            if not args.attention:
                batch_loss = u.apply_l1(model, batch_loss, args.l1)

            batch_loss.backward()
            #for name, p in model.named_parameters():
            #    print(name, np.linalg.norm(p.grad.data.numpy()))
            optimizer.step()
    else:
        loss, n, acc = u.eval_data(args, train, model)
        if not args.attention:
            loss = u.apply_l1(model, loss, args.l1)
        if args.cuda:   loss = loss.cuda()
        else:           loss = loss / n
        loss.backward()
        optimizer.step()

    if not args.attention:
        model.project_parameters() # project mixture parameters onto simplex
    model.eval()
    loss, n, acc_train = u.eval_data(args, train, model)
    nll_train = loss.cpu().data.numpy() #* n
    loss, n, acc_val = u.eval_data(args, val, model)
    nll_val = loss.cpu().data.numpy() #* n
    time_passed = time.time() - start_time
    if acc_train > 1. or acc_val > 1.:
        acc_train, acc_val = -1., -1.
    #if epoch % 400 == 0:
    #    print_params()
    if epoch % 1 == 0:
        print("Epoch: %d, NLL train: %.1f, NLL val: %.1f, Acc Train: %.2f, "
              "Acc Val: %.2f, %.2f s" %\
                (epoch, nll_train, nll_val, acc_train, acc_val, time_passed))

    if nll_val < best_val_loss: best_val_loss = nll_val
    if epoch > 400 and nll_val > 1.07*best_val_loss:
        print("Training stopped. Validation loss started increasing")
        break
    if epoch > 200 and nll_val < 0.15*u.uniform_loss(val):
        print("Training stopped. Converged!")
        break

loss, n, acc_test = u.eval_data(args, test, model)
nll_test = loss.cpu().data.numpy() #* n
print("Final. NLL Test: %.1f, Acc Test: %.2f" % (nll_test, acc_test))

if args.dir_to_save == '': dir_to_save = 'temp/%s' % model_name
else: dir_to_save = args.dir_to_save

model_path = dir_to_save + '/saved_model_' + str(args.seed)
torch.save(model.state_dict(), model_path)


if args.plot and args.attention and args.att_hid_layers <= 3:
    assert(model.training == False)
    from plot import save_att_plot, save_payoff_plot
    all_masks, att_vecs1, att_vecs2, att_out_vec = prep_for_plot(args,
                                                                 dir_to_save,
                                                                 test, model)
    for i in range(len(all_masks)):
        save_att_plot(all_masks[i][0], att_vecs1[i][0].flatten(),
                      att_vecs2[i][0].flatten(), i+1, dir_to_save, args.seed)
    save_att_out_plot(att_out_vec, dir_to_save, args.seed)


if not os.path.exists(dir_to_save):
    os.makedirs(dir_to_save)

options_dict = vars(args)
options_dict_name = '/options_dict_' + str(args.seed) + '.csv'
with open(dir_to_save + options_dict_name, 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in options_dict.items():
       writer.writerow([key, value])
    writer.writerow(['model_path', model_path])
    writer.writerow(['dir_to_save', dir_to_save])
    

result_filename = '/results%d.txt' % args.seed
result_test_filename = '/results_test.txt'
result_test_filename_acc = '/results_test_acc.txt'
with open(dir_to_save + result_filename, 'a') as f:
    f.write("Fold: %d, Epoch: %d, NLL train: %.1f, NLL val: %.1f, NLL test: %1.f,\
            Acc Train: %.2f, Acc Val: %.2f, Acc Test: %.2f, %.2f s\n" %\
                        (args.fold, epoch, nll_train, nll_val, nll_test,\
                         acc_train, acc_val, acc_test, time_passed))
with open(dir_to_save + result_test_filename, 'a') as f:
    f.write("%.2f\n" % nll_test)
with open(dir_to_save + result_test_filename_acc, 'a') as f:
    f.write("%.2f\n" % acc_test)

