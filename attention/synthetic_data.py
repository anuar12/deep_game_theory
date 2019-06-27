import math
import time

import numpy as np


""" 
Functions that generate synthetic games to test Attention layers.
"""


def test_game(payoffs, high, n_actions_square, n, m):
    ''' Several tests for generators of iterative elimination. 
        |Rationalizable set| > 1 is not supported. '''
    first_elimin_action = payoffs[0, -2, :n_actions_square] - \
                          payoffs[0, -1, :n_actions_square]
    square_elimin_action = payoffs[0, n_actions_square - 2, :n_actions_square] - \
                           payoffs[0, n_actions_square - 1, :n_actions_square]
    if np.any(payoffs < 0.) or not np.all(first_elimin_action > 0.):
        raise ValueError("Error: Last action should be dominated!")
    if not np.all(square_elimin_action > 0.):
        raise ValueError("Error: last action in the square matrix"
                         " should be dominated!")
    for i in range(0, n_actions_square - 1):
        for j in range(0, n_actions_square - 1):
            action_to_elimin = payoffs[0, i, :n_actions_square] - \
                               payoffs[0, j, :n_actions_square]
            if np.all(action_to_elimin > 0.) or np.all(action_to_elimin < 0.):
                msg = "Some action, apart from last, in row player shouldn't " \
                      "be dominated at the start!"
                raise ValueError(msg)
    for i in range(0, n_actions_square):
        for j in range(0, n_actions_square):
            action_to_elimin = payoffs[1, :n_actions_square, i] - \
                               payoffs[1, :n_actions_square, j]
            if np.all(action_to_elimin > 0.) or np.all(action_to_elimin < 0.):
                raise ValueError("Some action in col player shouldn't be"
                                 "dominated at the start!")
    if n > m:
        for i in range(m, n):
            action_to_elimin = payoffs[0, i - 1, :] - payoffs[0, i, :]
            if not np.all(action_to_elimin > 0.):
                raise ValueError(
                    'Error: row actions outside square should be dominated!')
    if n < m:
        for i in range(n, m):
            action_to_elimin = payoffs[1, :, n - 1] - payoffs[1, :, i]
            if not np.all(action_to_elimin > 0.):
                raise ValueError(
                    'Error: col actions outside square should be dominated!')
    if np.any(payoffs == 0.):
        raise ValueError("Error: Some matrix elements are not filled!")
    if n_actions_square > 2:
        first_action = payoffs[0, 0, :] - payoffs[0, 1, :]
        if np.all(first_action > 0.):
            raise ValueError("Error: First action shouldn't dominate second!")


def test_ration_set(payoffs, size_ration_set):
    ''' Make sure the actions in the rationalizable set are incomparable. '''
    if size_ration_set > 1:
        if payoffs[0, 0, 0] < payoffs[0, 1, 0]:
            msg = "Error: Row player can eliminate an action that's in ration. set!"
            raise ValueError(msg)
    for k in range(1, size_ration_set):
        if payoffs[0, k, k] < payoffs[0, k - 1, k]:
            msg = "Error: Row player can eliminate an action that's in ration. set!"
            raise ValueError(msg)
        if payoffs[1, k, k] < payoffs[1, k, k + 1]:
            msg = "Error: Col player can eliminate an action that's in ration. set!"
            raise ValueError(msg)


def shuffle_game(payoffs, action_counts, seed=123):
    ''' Shuffles across rows and columns. '''
    np.random.seed(seed)
    np.random.shuffle(payoffs[0, ...])  # shuffle along 1st player's actions
    np.random.seed(seed)
    np.random.shuffle(payoffs[1, ...])  # shuffle along 1st player's actions
    payoffs = np.swapaxes(payoffs, 1, 2)
    np.random.seed(seed + 1)
    np.random.shuffle(payoffs[0, ...])  # shuffle along 2nd player's actions
    np.random.seed(seed + 1)
    np.random.shuffle(payoffs[1, ...])  # shuffle along 2nd player's actions
    payoffs = np.swapaxes(payoffs, 1, 2)
    np.random.seed(seed)
    np.random.shuffle(action_counts)
    return payoffs, action_counts


def gen_rand_pl_payoff(n, m, high=300.):
    game = np.random.randint(high, size=n * m)
    game = game.reshape((n, m))
    return game


def gen_rand_game(n, m, high=300., is_uniform_labels=True):
    ''' Returns flattened game of random payoffs in range [0, high] and
    random labels '''
    payoffs = np.random.random((2 * n * m,)).reshape(1, -1) * high
    action_counts = np.zeros(n)
    if is_uniform_labels:
        action_counts[:] = 1.
        action_counts /= n
    else:
        action_counts[np.random.randint(n)] = 1.
    return payoffs, action_counts.reshape(1, -1)


def gen_dom_game(n, m, seed, is_second_pl=False, is_weak_dom=False, \
                 is_hard_labels=True, low_high=[200., 300.]):
    '''
    Generates a game of 2 players (2, n, m) shape where one action of 1st player
    dominates every other action. 2nd player payoffs are random.
    Hard labels mean all action counts go in the most dominated action,
    otherwise, it's weighted by amount of domination
    low_high = interval of payoffs of the dominating action
    UNSTABLE right now for n || m > 21
    Doesn't have iterative elimination.
    '''

    def gen_player_payoffs_actions(n, m, low_high):
        if n < 9 and m < 9:
            multiple = np.random.random() * 0.45 + 1.50  # range [1.5, 1.95]
        else:
            # to avoid numerical instability 
            multiple = np.random.random() * 0.1 + 1.50  # range [1.5, 1.60]

        def gen_action(i, low_high):
            if i == 0:
                new_low_high = low_high
            else:
                new_low_high = [low_high[0] / multiple ** i, low_high[0]]
            action = np.random.random(m) * (new_low_high[1] - new_low_high[0])
            action = action.astype(np.float64)
            action = action + new_low_high[0]
            return action, new_low_high

        player_payoffs = np.zeros((n, m))
        player_action_counts = np.zeros(n)

        for i in range(n):
            player_payoffs[i, :], low_high = gen_action(i, low_high)
            if i > 0 and is_weak_dom:
                num_weak_actions = np.random.randint(m)
                j = np.random.randint(m,
                                      size=num_weak_actions)  # take random action
                i_prev = np.random.randint(i - 1) if i > 1 else 0
                player_payoffs[i, j] = player_payoffs[i_prev, j]

        if is_hard_labels:
            player_action_counts[0] = 1.
        else:
            for i in range(n):
                i_action_count = int(1. / multiple ** i)
                player_action_counts[i] = i_action_count

        np.random.seed(seed)
        np.random.shuffle(player_payoffs)  # shuffle along 1st player's actions
        np.random.seed(seed)
        np.random.shuffle(player_action_counts)
        return player_payoffs, player_action_counts

    pl1, action_counts = gen_player_payoffs_actions(n, m, low_high)
    if is_second_pl:
        pl2, _ = gen_player_payoffs_actions(m, n, low_high)
        pl2 = pl2.swapaxes(0, 1)
    else:
        pl2 = gen_rand_pl_payoff(n, m)

    full_game_payoffs = np.concatenate((pl1[None, ...], pl2[None, ...]), 0)
    full_game_payoffs = full_game_payoffs.flatten()
    full_game_payoffs = full_game_payoffs.reshape(1, -1)
    action_counts = action_counts.reshape(1, -1)
    return full_game_payoffs, action_counts


def gen_iter_dom_game(n, m, seed, is_more_ration_actions=False, high=300.):
    '''
    Generates (2, n, m) game that produces a set of rationalizable actions
    through IEDS. A returned game has properties:
    - at every step in the IEDS, there is only a single action of a single
    player that is strictly dominated by some other action or actions.
    - will require (2*n - 2) iterations of IEDS
    - Resulting abs(n - m) actions are all trivially dominated
    - Is unstable for n || m > ~ 13
    '''
    if n < 9 and m < 9:
        mult = np.random.random() * 0.45 + 1.50  # [ 1.50, 1.95 ]
    else:
        mult = np.random.random() * 0.05 + 1.50  # [ 1.50, 1.55 ]
    payoffs = np.zeros((2, n, m))
    action_counts = np.zeros(n)

    # assume rationalizable action is (0, 0) index
    payoffs[0, 0, 0] = high - high / 4. + np.random.randint(0, high / 8)
    payoffs[1, 0, 0] = high - high / 4. + np.random.randint(0, high / 8)
    action_counts[0] = 1.

    # Populates the payoff array starting from the rationalizable action
    # at each step, the eliminated action is populated in the array, 
    # whose payoff is smaller than the previous step. There are (2*n - 2) 
    # iterations in this setup
    n_actions_square = np.min((n, m))
    for i in range(1, 2 * n_actions_square - 1):
        j = int(math.ceil(i / 2.))
        if i % 2 == 1:  # column player's turn
            val = high / mult ** j
            noise = np.random.random(j) * (val / 4)
            payoffs[1, :j, j] = high / mult ** j + noise
            noise = np.random.random(j - 1) * (
                        val / 4) if j > 0 and val > 2 else 0.
            payoffs[0, :j - 1, j] = high / mult ** j + noise
            payoffs[0, j - 1, j] = high / mult ** (j - 2) + \
                                   np.random.randint(0, val / 4) \
                if j == 0 else np.random.randint(3 * high / 4, high)
        else:  # row player's turn, j is incremented + 1
            val = high / mult ** j
            noise = np.random.random(j + 1) * (val / 4)
            payoffs[0, j, :j + 1] = high / mult ** j + noise
            noise = np.random.random(j) * (val / 4) if j > 0 and val > 2 else 0.
            payoffs[1, j, :j] = high / mult ** j + noise
            noise = np.random.random() * (val / 6) if j > 0 and val > 2 else 0.
            payoffs[1, j, j] = high / mult ** (j - 2) + noise

    payoffs[0, 0, 0] = payoffs[0, 1, 0] + np.random.random() * (high / 10)
    payoffs[0, 0, 1] = payoffs[0, 1, 1] + np.random.random() * (high / 10)
    payoffs[1, 0, 0] = payoffs[1, 0, 1] + np.random.random() * (high / 10)
    # Making sure that the max is not in the rationalizable set
    payoffs[0, n_actions_square - 2, n_actions_square - 1] = \
        high + np.random.randint(high / 8)

    # Finish populating payoffs outside of the square matrix
    # These actions must be dominated
    if n < 9 and m < 9:
        multiple = np.random.random() * 0.45 + 1.5
    else:
        multiple = np.random.random() * 0.05 + 1.5
    if n > m:
        for k in range(n_actions_square, n):
            payoffs[0, k, :] = payoffs[0, k - 1, :] / multiple
            payoffs[1, k, :] = np.random.randint(1, high, size=m)
    if n < m:
        for k in range(n_actions_square, m):
            payoffs[0, :, k] = np.random.randint(1, high, size=n)
            payoffs[1, :, k] = payoffs[1, :, k - 1] / multiple

    # let rationalizable set be of cardinality [1, min(n-1, m-1)]
    # Then we can add rationalizable actions by inserting one by one element
    # to make actions incomparable.
    if is_more_ration_actions:
        max_size_ration_set = np.min([n - 1, m - 1])
        if max_size_ration_set > 1:
            size_ration_set = np.random.randint(1, max_size_ration_set)
        else:
            size_ration_set = 1
        for i in range(1, size_ration_set):
            val = payoffs[0, i - 1, i]
            payoffs[0, i, i] = val + np.random.randint(val / 4)
            action_counts[i] = 1.
        action_counts /= float(size_ration_set)
        test_ration_set(payoffs, size_ration_set)

    if (not is_more_ration_actions) or (size_ration_set == 1):
        assert (payoffs[0, 0, 0] > payoffs[0, 1, 0] and \
                payoffs[0, 0, 1] > payoffs[0, 1, 1])

    # Randomize elements of the payoffs matrix which are eliminated by the
    # other player, by randomizing elements in the strictly upper/lower 
    # triangular matrix.
    if n_actions_square > 2:
        for i in range(2, n_actions_square):
            interval = [0., payoffs[0, i - 1, i]]
            payoffs[0, :i - 1, i] = np.random.random(i - 1) * interval[1]
            interval = [0., payoffs[1, i, i]]
            payoffs[1, i, :i] = np.random.random(i) * interval[1]

    test_game(payoffs, high, n_actions_square, n, m)
    payoffs, action_counts = shuffle_game(payoffs, action_counts, seed)
    payoffs = payoffs.flatten()
    payoffs = payoffs.reshape(1, -1)
    action_counts = action_counts.reshape(1, -1)
    return payoffs, action_counts


def gen_iter_dom_dec_game(n, m, seed, is_more_ration_actions=False, high=300.):
    ''' 
    Instead of using a multiple in the version above, here we decrement the 
    payoffs of the dominated action in each iteration. 
    '''
    payoffs = np.zeros((2, n, m))
    action_counts = np.zeros(n)

    # assume rationalizable action is (0, 0) index
    payoffs[0, 0, 0] = high - high / 4. + np.random.randint(0, high / 8)
    payoffs[1, 0, 0] = high - high / 4. + np.random.randint(0, high / 8)
    action_counts[0] = 1.

    # Populates the payoff array starting from the rationalizable action
    # at each step, the eliminated action is populated in the array, 
    # whose payoff is smaller than the previous step. There are (2*n - 2) 
    # iterations in this setup
    n_actions_square = np.min((n, m))
    max_dim = np.max([n, m])
    # make sure (constant*max_dim)<high cause we r going to subtract max_dim times
    # the values are in the range [some_val, some_val + c]
    c = np.random.random() * (high / (max_dim + 1))
    for i in range(1, 2 * n_actions_square - 1):
        j = int(math.ceil(i / 2.))
        if i % 2 == 1:  # column player's turn
            noise = np.random.random(j) * c - (
                        c / 2)  # zero mean, range=[-c/2, c/2]
            payoffs[1, :j, j] = high - j * c + noise
            noise = np.random.random(j - 1) * c - (c / 2)
            payoffs[0, :j - 1, j] = high - j * c + noise
            noise = np.random.random() * c - (c / 2)
            payoffs[0, j - 1, j] = high - (j - 2) * c
        else:  # row player's turn, j is incremented + 1
            noise = np.random.random(j + 1) * c - (c / 2)
            payoffs[0, j, :j + 1] = high - j * c + noise
            noise = np.random.random(j) * c - (c / 2)
            payoffs[1, j, :j] = high - j * c + noise
            noise = np.random.random() * c - (c / 2)
            payoffs[1, j, j] = high - (j - 2) * c + noise

    payoffs[0, 0, 0] = high + np.random.random() * c - (c / 2)
    payoffs[0, 0, 1] = high + np.random.random() * c - (c / 2)
    payoffs[1, 0, 0] = payoffs[1, 0, 1] + np.random.random() * c
    assert (payoffs[0, 0, 0] > payoffs[0, 1, 0] and
            payoffs[0, 0, 1] > payoffs[0, 1, 1])

    # Make the max not be in the rationalizable set, to make it difficult for the model
    payoffs[
        0, n_actions_square - 2, n_actions_square - 1] = high + np.random.random() * c

    # Finish populating payoffs outside of the square matrix
    # These actions must be dominated
    if n > m:
        for k in range(n_actions_square, n):
            noise = np.random.random(m) * c - (c / 2)
            payoffs[0, k, :] = payoffs[0, k - 1, :] - c + noise
            payoffs[1, k, :] = np.random.random(m) * high
    if n < m:
        for k in range(n_actions_square, m):
            payoffs[0, :, k] = np.random.random(n) * high
            noise = np.random.random(n) * c - (c / 2)
            payoffs[1, :, k] = payoffs[1, :, k - 1] - c + noise
    payoffs = payoffs.clip(min=0.) + 0.1

    # let rationalizable set be of cardinality in the range [1, min(n-1, m-1)]
    # Then we can add rationalizable actions by inserting one by one element
    # to make actions incomparable.
    if is_more_ration_actions:
        max_size_ration_set = np.min([n - 1, m - 1])
        if max_size_ration_set > 1:
            size_ration_set = np.random.randint(1, max_size_ration_set)
        else:
            size_ration_set = 1
        for i in range(1, size_ration_set):
            val = payoffs[0, i - 1, i]
            payoffs[0, i, i] = val + np.random.random() * (val / 4)
            action_counts[i] = 1.
        action_counts /= float(size_ration_set)
        test_ration_set(payoffs, size_ration_set)

    if (not is_more_ration_actions) or (size_ration_set == 1):
        assert (payoffs[0, 0, 0] > payoffs[0, 1, 0] and \
                payoffs[0, 0, 1] > payoffs[0, 1, 1])

    # Randomize elements of the payoffs matrix which are eliminated by the
    # other player, by randomizing elements in the strictly upper/lower 
    # triangular matrix.
    if n_actions_square > 2:
        for i in range(2, n_actions_square):
            interval = [0., payoffs[0, i - 1, i]]
            payoffs[0, :i - 1, i] = np.random.random(i - 1) * interval[1]
            interval = [0., payoffs[1, i, i]]
            payoffs[1, i, :i] = np.random.random(i) * interval[1]

    test_game(payoffs, high, n_actions_square, n, m)
    payoffs, action_counts = shuffle_game(payoffs, action_counts, seed)
    payoffs = payoffs.flatten()
    payoffs = payoffs.reshape(1, -1)
    action_counts = action_counts.reshape(1, -1)
    return payoffs, action_counts


def gen_many_dom_games(all_games={}, num_games=10, min_size=2, max_size=21,
                       is_second_pl=False, is_weak_dom=False,
                       is_hard_labels=True):
    for i in range(num_games):
        n = np.random.randint(min_size, max_size + 1)
        m = np.random.randint(min_size, max_size + 1)
        game_shape = (n, m)
        high = 1000.
        low_high = [100., high]
        seed = np.random.randint(1000)
        if game_shape in all_games:
            payoffs, action_counts = gen_dom_game(n, m, seed=seed,
                                                  is_second_pl=is_second_pl,
                                                  is_weak_dom=is_weak_dom,
                                                  is_hard_labels=is_hard_labels,
                                                  low_high=low_high)
            all_games[game_shape][0] = np.vstack((all_games[game_shape][0],
                                                  payoffs))
            all_games[game_shape][1] = np.vstack((all_games[game_shape][1],
                                                  action_counts))
        else:
            payoffs, action_counts = gen_dom_game(n, m, seed,
                                                  is_second_pl=is_second_pl,
                                                  is_weak_dom=is_weak_dom,
                                                  is_hard_labels=is_hard_labels,
                                                  low_high=low_high)
            all_games[game_shape] = [payoffs, action_counts]

    return all_games


def gen_many_iter_dom_games(all_games={}, num_games=5, min_size=2, max_size=5,
                            is_more_ration_actions=False, is_dec=False):
    for i in range(num_games):
        n = np.random.randint(min_size, max_size + 1)
        m = np.random.randint(min_size, max_size + 1)
        game_shape = (n, m)
        high = 1000.
        seed = np.random.randint(10000)
        if game_shape in all_games:
            if is_dec:
                payoffs, action_counts = gen_iter_dom_dec_game(n, m, seed,
                    is_more_ration_actions=is_more_ration_actions, high=high)
            else:
                payoffs, action_counts = gen_iter_dom_game(n, m, seed,
                    is_more_ration_actions=is_more_ration_actions, high=high)

            all_games[game_shape][0] = np.vstack((all_games[game_shape][0],
                                                  payoffs))
            all_games[game_shape][1] = np.vstack((all_games[game_shape][1],
                                                  action_counts))
        else:
            if is_dec:
                payoffs, action_counts = gen_iter_dom_dec_game(n, m, seed,
                    is_more_ration_actions=is_more_ration_actions, high=high)
            else:
                payoffs, action_counts = gen_iter_dom_game(n, m, seed,
                    is_more_ration_actions=is_more_ration_actions, high=high)
            all_games[game_shape] = [payoffs, action_counts]
    return all_games


def gen_all_games(num_games=100, min_size=2, max_size=5,
                  is_more_ration_actions=False):
    all_games = {}
    # all_games = gen_many_dom_games(all_games=all_games, num_games=num_games/3, min_size=min_size,\
    #                               max_size=max_size, is_weak_dom=False,\
    #                               is_hard_labels=True)
    all_games = gen_many_iter_dom_games(all_games=all_games,
                                        num_games=num_games / 2,
                                        min_size=min_size, max_size=max_size,
                                        is_more_ration_actions=is_more_ration_actions,
                                        is_dec=False)
    all_games = gen_many_iter_dom_games(all_games=all_games,
                                        num_games=num_games / 2,
                                        min_size=min_size, max_size=max_size,
                                        is_more_ration_actions=is_more_ration_actions,
                                        is_dec=True)
    return all_games


def gen_many_rand_games(num_games=5, min_size=2, max_size=5):
    all_games = {}
    for i in range(num_games):
        n = np.random.randint(min_size, max_size + 1)
        m = np.random.randint(min_size, max_size + 1)
        game_shape = (n, m)
        seed = np.random.randint(time.time())
        if game_shape in all_games:
            payoffs, action_counts = gen_rand_game(n, m, high=300.)
            all_games[game_shape][0] = np.vstack((all_games[game_shape][0],
                                                  payoffs))
            all_games[game_shape][1] = np.vstack((all_games[game_shape][1],
                                                  action_counts))
        else:
            payoffs, action_counts = gen_rand_game(n, m, high=300.)
            all_games[game_shape] = [payoffs, action_counts]
    return all_games


''' Manual 2-iteration Game for Testing '''


def get_double_iter_dom_game(n=4, m=4):
    payoffs = np.zeros((2, n, m))
    action_counts = np.zeros(n)
    action_counts[0] = 1.
    payoffs[0, 0, :2] = 300.
    payoffs[0, 1, :2] = 100.
    payoffs[0, 0, 2:] = 100.
    payoffs[0, 1, 2:] = 500.
    payoffs[0, 2:, :] = 10.

    payoffs[1, :2, :2] = np.random.random((2, 2)) * 900 + 100.
    payoffs[1, :2, 2:] = np.random.random((2, 2)) * 100.
    payoffs[1, 2:, :] = np.random.random((2, 4)) * 1000.
    print payoffs

    payoffs, action_counts = shuffle_game(payoffs, action_counts, seed)
    payoffs = payoffs.flatten().reshape(1, -1)
    action_counts = action_counts.reshape(1, -1)
    return payoffs, action_counts


''' Custom difficult games '''

iter_game1 = np.array(
    [[[10., 20, 100],
      [20, 30, 10],  # second is IEDS action
      [1, 1, 1]],
     [[10, 2000, 1],
      [10, 20, 1],
      [10, 20, 1]]])
game1_labels = np.array([0., 10., 0.]).reshape(1, -1)

iter_game2 = np.array(
    [[[20., 100, 40],
      [30, 10, 30],  # first is IEDS action
      [10, 100, 20],
      [1, 1, 1]],
     [[20, 1, 10],
      [50, 1, 40],
      [40, 20, 30],
      [10, 1, 50]]])
game2_labels = np.array([0., 10., 0., 0.]).reshape(1, -1)

iter_game3 = np.array(
    [[[10., 40, 30],
      [20, 30, 50],
      [30, 50, 40]],
     [[30, 20, 40],
      [50, 20, 20],
      [20, 50, 30]]])
game3_labels = np.array([0., 0., 10.]).reshape(1, -1)

iter_game4 = np.array(
    [[[10., 10, 10, 10, 30, 30, 30, 30],
      [30, 20, 30, 20, 30, 40, 30, 40],
      [10, 10, 20, 20, 10, 10, 20, 20]],
     [[30, 30, 30, 30, 10, 10, 10, 10],
      [10, 30, 0., 20, 10, 20, 10, 20],
      [30, 40, 10, 10, 10, 20, 10, 10]]])  # second is IEDS
game4_labels = np.array([0., 0., 10]).reshape(1, -1)

iter_game_5x5 = np.array(
    [[[4, 3, -3, -1, -2],
      [-1, 2, 2, -1, 2],
      [2, -1, 0, 4, 0],
      [1, -3, -1, 1, -1],
      [0, 1, -3, -2, -1]],

     [[-1, 0, 1, 4, 0],
      [1, 2, 3, 0, 5],
      [1, -1, 4, -1, 2],
      [6, 0, 4, 1, 4],
      [0, 4, 1, 3, -1]]], dtype=np.float32)
iter_game_5x5 += 4.
iter_game_5x5 *= 100.
game_5x5_labels = np.array([0., 1., 0., 0., 0.]).reshape(1, -1)

iterated_games = {(4, 3): [iter_game2.flatten(), game2_labels], \
                  (3, 3): [np.concatenate((iter_game1.flatten()[None, :], \
                                           iter_game3.flatten()[None, :]),
                                          axis=0), \
                           np.concatenate((game1_labels, game3_labels),
                                          axis=0)],
                  (3, 8): [iter_game4.flatten(), game4_labels]}
