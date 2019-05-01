from __future__ import print_function

import subprocess
import itertools
import argparse
import sys
import os
from time import sleep
import random
import stat

parser = argparse.ArgumentParser()
parser.add_argument('-max_jobs',default=1,type=int)
parser.add_argument('-output_dir',default='../logs_torch',type=str)
parser.add_argument('-output_file',default='all_jobs_torch.sh',type=str)
args = parser.parse_args(sys.argv[1:])

slurm_cmd = 'python semantic_pytorch.py --output_dir {}'.format(args.output_dir) 


gpu = [0]
num_labeled = [1000]
batch_size = [10,16]
std = [0.2, 0.3, 0.4]
#wt = [2,1,0.1, 0.01, 0.005]
wt = [0]
lr = [1e-4]
num_iter = [25000]

all_params = [num_labeled,  batch_size, std, wt, lr, gpu, num_iter]
names = ['num_labeled','batch_size','std','wt','lr', 'gpu','num_iter']

all_jobs = list(itertools.product(*all_params))
jobs_list = {}
for i, setting in enumerate(all_jobs):
    name_setting = {n: s for n, s in zip(names, setting)}
    setting_list = ['--%s %s' % (name, str(value)) for name, value in name_setting.items()]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (n.replace('_','.'), str(s)) for n, s in name_setting.items()])
    jobs_list[log_str] = setting_str


fh = open(args.output_file,'w')
mode = stat.S_IROTH | stat.S_IRWXU | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP

print('Running %d jobs' % (len(jobs_list)))
count = 0
max_jobs = args.max_jobs
for log_str, setting_str in jobs_list.items():
    full_cmd = '{} {}&'.format(slurm_cmd, setting_str)
    print(full_cmd,file =fh)
    print("pids[{}]=$!".format(count%max_jobs),file = fh)
    if count % max_jobs == (max_jobs -1):
        print("echo 'job count is {}'".format(count), file=fh)
        print("for pid in ${pids[*]}; do",file = fh)
        print("\twait $pid",file = fh)
        print("done",file = fh)
    #
    count += 1

print("for pid in ${pids[*]}; do",file = fh)
print("\twait $pid",file = fh)
print("done",file = fh)
fh.close()
os.chmod(args.output_file,mode)


