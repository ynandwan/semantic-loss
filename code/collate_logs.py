import itertools
import argparse
import sys
import os
from time import sleep
import random
import stat
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-logs_dir',default='../logs_colab',type=str)
args = parser.parse_args(sys.argv[1:])

def read_table(stats_file):
    table = None
    try:
        table = pd.read_csv(stats_file)
    except:
        pass
    #
    if table is not None:
        for (colname,val) in [('-'.join(x.split('-')[:-1]),x.split('-')[-1]) for x in table.exp[0].split('_')]:
            table[colname] = val
    #
    return table


table = None 
explist = os.listdir(args.logs_dir)
missing = []
for exp in explist:
    stats_file = os.path.join(args.logs_dir,exp)
    this_table = read_table(stats_file)
    if table is None:
        table = this_table
    elif this_table is not None:
        table = table.append(this_table)
    else:
        missing.append(exp)




table = table.reset_index(drop=True)
results = table.loc[table.groupby('exp')['tea'].idxmax()]

