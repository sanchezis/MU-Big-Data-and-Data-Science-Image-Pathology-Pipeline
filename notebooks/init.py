import logging
import sys 
import os
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython.display import display_html

# logging.warning(f"{sys.executable}")

os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'
os.environ['PYARROW_IGNORE_TIMEZONE']='1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if './../lib' not in os.environ.get('PATH', '') or \
    os.path.join(  os.getcwd() , 'lib'  ) not in os.environ.get('PATH', ''):
    os.environ['PATH'] += os.pathsep + './../lib' + os.pathsep + '../lib' + os.pathsep + os.path.join( os.path.dirname(  os.getcwd() ), 'lib'  )
    os.environ['PYTHONPATH'] = sys.executable
    
sys.path += [ f"./.." , f"./../digital_pathology" , './../lib', os.path.join( os.path.dirname(  os.getcwd() ), 'digital_pathology'  )]

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

style = [e for e in plt.style.available if 'talk' in e.lower()][0]
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
