import pandas as pd 
import os.path as op 

path = op.join(op.dirname(__file__), 'june_2021.h5')
dataset = pd.read_hdf(path, key='df')

# Both UK and GB are used, modify in-place so all are UK
dataset.loc[dataset['region'] == 'GB', 'region'] = 'UK'