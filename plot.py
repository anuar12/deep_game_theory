import csv
import pprint as pp

import torch
import numpy as np
import pandas as pd
import plotly
from plotly.offline import download_plotlyjs, plot
from plotly.graph_objs import *
import matplotlib.pyplot as plt

from attention.synthetic_data import gen_many_iter_dom_games, iter_game_5x5, game_5x5_labels
from attention.attend import AttentionNet
import utils as u


''' Saves html files of attention plots at each level '''


is_5x5_test_game = True    # predicts and plots on the 5x5 game from the slides


def make_anno(x=1, y=1, text='_citation_'):
    return Annotation(
        text=text,          # annotation text
        showarrow=False,    # remove arrow 
        xref='paper',     # use paper coords
        yref='paper',     #  for both coordinates
        xanchor='right',  # x-coord line up with right end of text 
        yanchor='bottom', # y-coord line up with bottom end of text 
        x=x,              # position's x-coord
        y=y               #   and y-coord
    )

def add_att_layer_to_data(data, mask, att_vec1, att_vec2):
    n, m = mask.shape
    x = np.arange(m) + 1
    y = np.arange(n) + 1
    mask_heatmap = Heatmap(z=mask, x=x, y=y,
                      xaxis='x1',
                      yaxis='y1',
                      colorbar=dict(tick0=0., dtick=0.05, ticks='outside'))
    att_col_bar = Bar(x=x,
                 y=att_vec2,
                 xaxis='x1',
                 yaxis='y2')
    att_row_bar = Bar(x=att_vec1,
                 y=y,
                 xaxis='x2',
                 yaxis='y1',
                 orientation='h')
    data.extend([mask_heatmap, att_col_bar, att_row_bar])
    return data

def update_att_layer_to_layout(layout, n_layer):
    x_title = 'Column Player' 
    y_title = 'Row Player'
    title = 'Attention in layer %d' % (n_layer)

    layout.update(
        title=title,  # set plot's title
        font=Font(
            family='PT Sans Narrow, sans-serif',  # global font
            size=13
        ),
        xaxis=XAxis(
            title=x_title,   # set x-axis title
            dtick=1.,
            zeroline=False,   # remove x=0 line
            domain=[0., 0.7]
        ),
        yaxis=YAxis(
            title=y_title,   # y-axis title
            dtick=1.,
            zeroline=False,   # remove y=0 line
            domain=[0., 0.7],
            showgrid=False
        ),
        annotations=Annotations([  # add annotation citing the data source
            make_anno()
        ]),

        xaxis2=XAxis(
            domain=[0.75, 1.], # domain of x-axis2
            zeroline=False,   # remove x=0 line
            showgrid=True,     # show horizontal grid line,
            dtick=0.2,
            range=[0., 1.05]
        ),

        yaxis2=YAxis(
            domain=[0.75, 1.], # domain of y-axis2
            zeroline=False,   # remove y=0 line
            showgrid=True,     # show vertical line
            dtick=0.2,
            range=[0., 1.05]
        ),

        showlegend=False,  # remove legend
        autosize=False,    # custom size
        width=800,         # set figure width 
        height=650,         #  and height   
    )
    return layout


def get_plotly_fig(mask, att_vec1, att_vec2, n_layer):
    data = add_att_layer_to_data(Data([]), mask, att_vec1, att_vec2)
    layout = update_att_layer_to_layout(Layout([]), n_layer)
    fig = Figure(data=data, layout=layout)
    return fig


import io
import pandas as pd
import plotly
import plotly.graph_objs as go
import selenium.webdriver as webdriver
import shutil
import time

from os.path import devnull
from PIL import Image
from subprocess import Popen, PIPE

        
### from bokeh/util, slightly modified to avoid using bokeh's settings.py
### - https://github.com/bokeh/bokeh/blob/master/bokeh/util/dependencies.py
def detect_phantomjs():
    '''Detect if PhantomJS is avaiable in PATH.'''
    try:
        phantomjs_path = shutil.which('phantomjs')
    # Python 2 relies on Environment variable in PATH - attempt to use as follows
    except AttributeError:
        phantomjs_path = "phantomjs"

    try:
        proc = Popen([phantomjs_path, "--version"], stdout=PIPE, stderr=PIPE)
        proc.wait()

    except OSError:
        raise RuntimeError('PhantomJS is not present in PATH. Try "conda install phantomjs" or \
            "npm install -g phantomjs-prebuilt"')
    return phantomjs_path
       

### from bokeh/io, slightly modified to avoid their import_required util
### - https://github.com/bokeh/bokeh/blob/master/bokeh/io/export.py
def create_default_webdriver():
    '''Return phantomjs enabled webdriver'''
    phantomjs_path = detect_phantomjs()
    return webdriver.PhantomJS(executable_path=phantomjs_path, service_log_path=devnull)

def get_payoff_plot(payoffs, pl):
    df = pd.DataFrame(payoffs)
    df.insert(0, 'num_action', range(1, df.shape[0]+1))
    trace1 = Table(header=dict(values=['num_action'] + [int(i)+1 for i in df.columns[1:]]),
                  cells=dict(
                        values=[df.iloc[:, i] for i in range(len(df.columns))]))
    #data = [trace1]
    #layout = dict(title='Player %d' % pl, width=400, height=300)
    #fig = dict(data=data)#, layout=layout)
    some_fig = Figure(data=Data([trace1]))
    plotly.offline.plot(some_fig, filename='payoffs'+str(pl)+'.html', auto_open=False)

def save_payoff_plot(payoffs, action_counts, game_shape):
    n, m = game_shape
    payoffs = payoffs.reshape(2, n, m)
    #get_payoff_plot(payoffs[0, ...], 1)
    #get_payoff_plot(payoffs[1, ...], 2)


def prep_for_plot(args, dir_to_save, test, model):
    game_shape = (args['min_size'], args['max_size'])
    game_shape = (5, 5)
    game_to_plot = {game_shape: test[game_shape]}
    print "game_to_plot: ", game_to_plot
    np.save(dir_to_save + '/payoffs', test[game_shape][0])
    np.save(dir_to_save + '/action_counts', test[game_shape][1])
    model.eval()
    u.eval_data(args, game_to_plot, model)  # implicitly saves masks and vecs in model object
    payoffs, action_counts = game_to_plot[game_shape]
    print "Payoffs: \n", payoffs.reshape(2, game_shape[0], game_shape[1])
    print "Labels: ", action_counts
    all_masks = [model.all_masks[i].data.numpy() for i in range(len(model.all_masks))]
    att_vecs1 = [model.att_vecs[0][i].data.numpy() for i in range(len(model.att_vecs[0]))]
    att_vecs2 = [model.att_vecs[1][i].data.numpy() for i in range(len(model.att_vecs[1]))]
    print "Output vector: ", model.out_att_vec
    att_out_vec = model.out_att_vec.data.numpy()[0, ...]
    np.save(dir_to_save + '/out_vec', model.out_att_vec)
    save_payoff_plot(payoffs[0, ...], action_counts[0, ...], game_shape)
    return all_masks, att_vecs1, att_vecs2, att_out_vec


def save_att_plot(mask, att_vec1, att_vec2, n_layer, dir_to_save, seed):
    fig = get_plotly_fig(mask, att_vec1, att_vec2, n_layer)
    filename = dir_to_save + '/att_plot' + str(seed) + str(n_layer)
    plotly.offline.plot(fig, filename=filename+'.html', auto_open=False)

    ### create webdrive, open file, maximize, and sleep
    #driver = create_default_webdriver()
    #driver.get(filename+'.html')
    #driver.maximize_window()
    #time.sleep(1)

    #png = driver.get_screenshot_as_file(filename+'.png')
    #image = Image.open(filename+'.png')
    #driver.quit()
    print "Finished plotting layer %d!" % n_layer

def save_att_out_plot(att_out_vec, dir_to_save, seed):
    filename = dir_to_save + '/att_plot' + str(seed)
    x = np.arange(att_out_vec.shape[0]) + 1
    att_out_bar = Bar(x=x,
                      y=att_out_vec)
    layout_out = Layout(title='Output Layer',
                        xaxis=dict(dtick=1),
                        yaxis=dict(range=[0, 1]))
    fig_out = Figure(data=Data([att_out_bar]), layout=layout_out)
    plotly.offline.plot(fig_out, filename=filename+'_out.html', auto_open=False)


def predict_with_model(path_to_options):
    with open(path_to_options, 'rb') as csv_file:
	reader = csv.reader(csv_file)
	args_pre = dict(reader)
    args = {}
    for key, val in args_pre.items():
        try:
            args[key] = eval(val)
        except:
            args[key] = val
    model = AttentionNet(hid_layers=args['att_hid_layers'], hid_units=args['att_hid_units'],\
                         is_simult=False, is_fc_first=False, is_fc_hid=False,\
                         with_last=True, is_cuda=args['cuda'], drop_p=0.) 
    model.load_state_dict(torch.load(args['model_path']))
    game = gen_many_iter_dom_games(all_games={}, num_games=1, min_size=args['min_size'],\
                               max_size=args['max_size'], is_more_ration_actions=False,
                               is_dec=True)
    if is_5x5_test_game:
        game_5x5 = iter_game_5x5
        game_5x5 = {(5, 5): [iter_game_5x5.flatten().reshape(1, -1), game_5x5_labels]}
        game = game_5x5
    all_masks, att_vecs1, att_vecs2, att_out_vec = prep_for_plot(args, args['dir_to_save'], game, model)
    for i in range(len(all_masks)):
        save_att_plot(all_masks[i][0], att_vecs1[i][0].flatten(), att_vecs2[i][0].flatten(), i+1, args['dir_to_save'], args['seed'])
    save_att_out_plot(att_out_vec, args['dir_to_save'], args['seed'])
    
    
if __name__ == '__main__':
    path_to_options = 'out_prod_first/attention/1000g_iter_hard_5hl_2hu_lr0.0005_7x7/options_dict_100.csv'
    path_to_options = 'temp/options_dict_101.csv'
    path_to_options = 'out/attention/1000g_all_4hl_2hu_lr0.0010_2x6/options_dict_118.csv'
    predict_with_model(path_to_options)





