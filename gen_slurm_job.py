import os
import datetime.datetime as dt
import csv


""" Generate SLURM scheduler job batch script """


def parse_options(path_to_options):
    with open(path_to_options, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        args_pre = dict(reader)
    args = {}
    for key, val in args_pre.items():
        try:
            args[key] = eval(val)
        except:
            args[key] = val 
    return args

path_to_options = ''

if path_to_options == '':
    lr = 1e-3
    epochs = 5000
    model_name = 'attention'
    synthetic_data = 'all' 
    iters = 10
    att_hid_units = 2
    att_hid_layers = 4
    min_size = 5
    max_size = 5
    is_plot = False
    is_cuda = False
    if is_cuda: cuda_string = '\n#SBATCH --gres=gpu:1'
    else: cuda_string = ''
    is_more_ration_actions = False
else:
    for k, v in args.items():
        exec(k + " = " + v)

dir_to_save = 'out/%s/2500g_%s_%dhl_%dhu_lr%.4f_%dx%d' % (model_name, synthetic_data,\
                                                    att_hid_layers, att_hid_units,\
                                                    lr, min_size, max_size)
curr_time = str(dt.now())[:-10]
dir_to_save += curr_time

if not os.path.exists(dir_to_save):
    os.makedirs(dir_to_save)

if model_name == 'attention' and ('iter' in synthetic_data or synthetic_data == 'simple'\
                                  or synthetic_data == 'all'):
    iters = iters 
    command = '''#!/bin/bash

#SBATCH --job-name=%s_%dx%d
#SBATCH --output=%s/job_%%A_%%a.out
#SBATCH --error=%s/job_%%A_%%a.err
#SBATCH --array=0-%d
#SBATCH --ntasks=1%s
#SBATCH --mem=8G
#SBATCH --time=0-02:00
#SBATCH --account=def-kevinlb


seedVal=`expr $SLURM_ARRAY_TASK_ID + 100`
python train.py --%s True --epochs %d --lr %f --seed $seedVal --att_hid_units %d --att_hid_layers %d --synthetic_data %s --min_size %d --max_size %d --dir_to_save %s --batch_size 64 ''' % (model_name, min_size, max_size, dir_to_save, dir_to_save, iters, cuda_string, model_name, epochs, lr,\
           att_hid_units, att_hid_layers, synthetic_data, min_size, max_size, dir_to_save)
else:
    cv_iters = 1
    array_jobs = cv_iters*10 - 1 
    command = '''#!/bin/bash

#SBATCH --job-name=%s
#SBATCH --output=%s/job_%%A_%%a.out
#SBATCH --error=%s/job_%%A_%%a.err
#SBATCH --array=0-%d
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=0-02:00
#SBATCH --account=def-kevinlb


foldVal=`expr $SLURM_ARRAY_TASK_ID %% 10`
seedVal=`expr $SLURM_ARRAY_TASK_ID / 10 + 111`
python train.py --fold $foldVal --%s True --epochs %d --lr %f --seed $seedVal --att_hid_units %d --att_hid_layers %d --dir_to_save %s --batch_size 32 ''' % (model_name, dir_to_save, dir_to_save, array_jobs, model_name, epochs, lr,\
           att_hid_units, att_hid_layers, dir_to_save)

if is_plot: command += ' --plot'
if not is_cuda: command += ' --disable_cuda'
if is_more_ration_actions: command += ' --is_more_ration_actions'

with open('runner.sl', 'w+') as f:
    f.write(command)


