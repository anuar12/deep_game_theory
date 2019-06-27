import os

import pandas as pd


''' Calculates mean and std of test loss and accuracy ''' 

def get_stats(df):
    return df.sum() / df.shape[0], df.std(), df.min(), df.max()

path_to_experiments = '/home/anuar12/DeepCH-pytorch/out/attention'
for experiment in os.listdir(path_to_experiments):
    experiment_folder = path_to_experiments + '/' + experiment
    print "\nExperiment: ", experiment

    filename = experiment_folder + '/results_test.txt'
    filename_acc = experiment_folder + '/results_test_acc.txt'

    try:
        df = pd.read_csv(filename, squeeze=True)

        print "Number of iters: ", df.shape
        print "Loss Mean: %.2f, Std: %.2f, Best: %.2f, %.2f" % get_stats(df)

        df = pd.read_csv(filename_acc)
        print "Accuracy Mean: %.3f, Std: %.2f, Best: %.2f, %.2f" % get_stats(df)
    except IOError:
        print "======= Error! Perhaps no file found ======="
