import argparse
import time

import numpy as np
try:
    import bogota.datapool
    from bogota.utils import action_profiles
except:
    'Bogota import failed...'


class GameData(object):
    def __init__(self, filename=None, normalize=1.):
        # data = { (shape) : [[payoffs1], [actioncounts1],
        #                     [payoffs2], [actioncounts2]] }
        self._data = {}
        if filename is not None:
            self.read_csv(filename, normalize)
        
    def add_game(self, payoffs, actioncounts, shape):
        if shape in self._data:
            self._data[shape][0] = np.vstack((self._data[shape][0], payoffs))
            self._data[shape][1] = np.vstack((self._data[shape][1], actioncounts))
        else:
            payoffs = payoffs.reshape((1, payoffs.shape[0]))
            actioncounts = actioncounts.reshape((1, actioncounts.shape[0]))
            self._data[shape] = [payoffs, actioncounts]
    
    def datalist(self):
        return self._data

    def read_csv(self, filename, normalize=1.):
        with open(filename) as f:
            for i, line in enumerate(f):
                if i ==0:
                    continue
                else:
                    title, shape, payoffs, actioncounts = line.split('\t')
                    shape = tuple(eval(shape))
                    payoffs = np.array(eval(payoffs)) / float(normalize)
                    actioncounts = np.array(eval(actioncounts))
                    self.add_game(payoffs, actioncounts, shape)
    
    def _generate_game_dictionary(self):
        game_dict = {}
        for shape, games in self._data.iteritems():
            for payoff, actioncounts in zip(games[0], games[1]):
                idx = tuple(payoff)

                if idx in game_dict:
                    game_dict[idx] = (shape, game_dict[idx][1] + actioncounts)
                else:
                    game_dict[idx] = (shape, actioncounts)
        return game_dict
    
    def _split_indices_into_folds(self, indices, num_folds):
        n = len(indices)
        s = n/num_folds
        fold_indices = [list(indices)[i*s:(i+1)*s] for i in range(num_folds)] 
        for idx, i in enumerate(list(indices)[s*num_folds:]):
            fold_indices[idx] += [i]
        return fold_indices

    def train_test(self, fold, num_folds=10, seed=123):
        rng = np.random.RandomState(seed)
        gd = self._generate_game_dictionary() # generated dict of {[payoffs]: ([shape], [actions])}
        game_idx = gd.keys()              # list of all payoffs
        indices = range(len(gd.keys()))   # range of the number of games in game_dict
        rng.shuffle(indices)              # check whether seed is different
        fold_indices = self._split_indices_into_folds(indices, num_folds)
        #print "INDS:", fold_indices
        #print "INDICES: ", len(indices)
        #print "FOLD INDICES: ", len(fold_indices)
        train = GameData()
        test = GameData()
        print "fold: ", fold
        for i, fold_idx in enumerate(fold_indices):
            for idx in fold_idx:
                shape, actioncount = gd[game_idx[idx]]
                if i != fold:
                    train.add_game(np.array(game_idx[idx]), actioncount, shape)
                else:
                    test.add_game(np.array(game_idx[idx]), actioncount, shape)
        return train, test

    def are_large_games_present(self):
        for item in self._data.keys():
            print item,
            if item == (121, 121):
                print "121 Game present!"
            if item == (61, 61):
                print "61 Game present!"


def kfold(fold_function, start_fold=0, end_fold=10):
    for i in xrange(start_fold, end_fold):
        i = i % 10
        print 'STARTING FOLD: %d' % (i + 1)
        t = time.time()
        costs = fold_function(i)
        t = time.time() - t
        print 'Fold %d complete in %f seconds' % (i + 1 , t)
        print '*' * 100
        return costs


def parse_args():
    parser = argparse.ArgumentParser(description="K-fold cross validation")
    parser.add_argument('--start_fold', default=0, type=int)
    parser.add_argument('--end_fold', default=10, type=int)
    parser.add_argument('--path', default='')
    parser.add_argument('--json', default=None,
        help="Path of json file describing the options of the experiment")
    return parser.parse_args()


def build_fold_function(options, new_experiment=False, resume=False):
    # if not os.path.exists("./test/best_loss"):
    #     os.makedirs("./test/best_loss")
    # output_file = options.get('path', './') + options.get('name', 'test') + '.csv'
    # best_loss_filename = options.get('path', './') + "best_loss/" + options.get('name', 'test') + '.csv'
    par_file = options.get('path', './') + options.get('name', 'test') + '_%d_par.json'
    dataset_name = options.get('dataset', 'all9')
    seed = options.get('model_seed', 123)   # OR SEED!
    # if not os.path.isfile(output_file):
    #     with open(output_file, 'w') as f:
    #         f.write('Data: %s, seed: %d\n' % (dataset_name, seed))
    #         f.write(','.join(['fold', 'seed', 'train', 'valid', 'test']) + '\n')

    def fold_function(k):  # k is the fold index
#         data = GameData('./all9small.csv', 50.)   # don't include 62 and 121
        data = GameData('./all9.csv', 50.)
        print "Shapes of games: ", data.datalist().keys()
        train_data, test_data = data.train_test(k, seed=seed)
        return data

    return fold_function


def log_fold(log_file_name, llk, fold, model_seed, llk_start=None):
    log_file_name = log_file_name.replace(".csv", "_out.csv")
    with open(log_file_name, 'a') as f:
        if llk_start is not None:
            log = [fold] + [model_seed] + list(llk) + list(llk_start)
        else:
            log = [fold] + [model_seed] + list(llk)
        # f.write(','.join(str_lst(log)) + '\n')


def str_lst(x):
    return [str(i) for i in x]



DEFAULT_OPTIONS = {'name': 'test',
                   'save_path': './',
                   'hidden_units': [50, 50],
                   'activ': 'relu',
                   'pooling': True,
                   'batch_size': None,
                   'ar_layers': 15,
                   'dropout': False,
                   'l1': 0.01,
                   'l2': 0.0,
                   'pooling_activ':'max',
                   'opt': 'adam',
                   'max_itr': 3,
                   'model_seed': 3,
                   'objective': 'nll'}


def get_data():
    options = DEFAULT_OPTIONS
    print "OPTIONS FROM JSON LOADED SUCCESSFULLY."

    fold_function = build_fold_function(options, 0, False)
    data = fold_function(0).datalist()
    return data
