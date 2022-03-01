
import os.path
import numpy as np

def read_configuration(file_path):
	config = {}
	if not os.path.exists(file_path):
		print('config file is not found!')
		raise IOError
	with open(file_path) as f:
		for ind,line in enumerate(f):
			if line.strip()!='':
				try:
					key,value=line.strip().split('=')
					config[key]=value
				except ValueError:
					print('config file is not in the correct format! Error Line:%d'%(ind))
	return config

def maxminnorm_matrix(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t


def maxminnorm_list(array):
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    t=np.empty((data_rows))
    t=(array-mincols)/(maxcols-mincols)
    return t

