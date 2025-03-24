# Copyright (c) 2024 Israel Llorens
# Licensed under the EUPL-1.2  

__author__ = "Israel Llorens <sanchezis@hotmail.com>"
__copyright__ = "Copyright 2024, Israel Llorens"
__license__ = "EUPL-1.2"

import sys 
import os
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

os.environ.setdefault('HADOOP_CONF_DIR', '/etc/hadoop/conf')
os.environ.setdefault('PYARROW_IGNORE_TIMEZONE', '1')
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if './../lib' not in os.environ.get('PATH', '') or \
    os.path.join(  os.getcwd() , 'lib'  ) not in os.environ.get('PATH', ''):
    os.environ['PYTHONPATH'] = sys.executable
    os.environ['LIBPATH'] = os.pathsep + './lib' \
                        + os.pathsep + './../lib' \
                        + os.pathsep + '../lib' \
                        + os.pathsep + os.path.join(   os.getcwd() , 'lib'  ) \
                        + os.pathsep + os.path.join( os.path.dirname(  os.getcwd() ), 'lib'  ) 
    os.environ['DYLD_LIBRARY_PATH'] = os.environ['LIBPATH']
    os.environ['LIBRARY_PATH'] = os.environ['LIBPATH']

#Ipython notebook settings
pd.set_option('max_info_columns', 500)
pd.set_option('max_colwidth',800)

pd.set_option('large_repr', 'info')
pd.set_option ('display.notebook_repr_html', True)

pd.set_option('expand_frame_repr', True)
#pd.set_option('display.html.table_schema', True)
pd.set_option('max_info_rows', 500)

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 2000)  # Uncomment this line to show all text in column

pd.set_option('display.large_repr', 'truncate')
pd.options.display.float_format = '{:,.2f}'.format

style = next((e for e in plt.style.available if 'talk' in e.lower()), 'default')
plt.style.use(style)

palette = ['#0079B1', '#009FDB', '#00C9FF']

mpl.rcParams['figure.figsize'] = (28,8)
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams["axes.grid"] = True
mpl.rcParams['grid.color'] = "grey"
mpl.rcParams['grid.alpha'] = .1
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams["axes.grid.axis"] ="y"
mpl.rcParams["figure.facecolor"] = 'white'
mpl.rcParams["axes.facecolor"] = 'white'
mpl.rcParams["savefig.facecolor"] = 'white'

# mpl.rcParams['font.size'] =  32
# mpl.rcParams['legend.fontsize'] = 22
mpl.rcParams['figure.dpi'] = 256//4

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
