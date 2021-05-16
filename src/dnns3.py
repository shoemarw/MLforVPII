import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

""" Data prep functions
"""

def prepDataFrame(filename, k=4):
    """ Read in the loads and stores from a trace and 
    """

    # Read in the data and get it ready for computation
    df = pd.read_csv(filename)
    df = df.replace('NAN', np.nan).dropna()
    #df = df.astype('int64')
    df['pc'] = df['pc'].astype('int64')
    df['effective_address'] = df['effective_address'].astype('int64')
    df['val0'] = df['val0'].astype('int64')
    df['store-1val0'] = df['store-1val0'].astype('int64')
    df['store-2val0'] = df['store-2val0'].astype('int64')
    df['store-1ea'] = df['store-1ea'].astype('int64')
    df['store-2ea'] = df['store-2ea'].astype('int64')

    # Build a dataframe with the 64bit values split into 16 bit values

    # pc 
    df1 = pd.DataFrame(index=df.index)
    df1['pc-00-15'] = df['pc'] % math.pow(2, 15)
    df['pc-top']    = df['pc'] - df1['pc-00-15']
    df1['pc-16-31'] = df['pc-top'] % math.pow(2, 31)
    df['pc-top']    = df['pc-top'] - df1['pc-16-31']
#    df1['pc-32-47'] = df['pc-top'] % math.pow(2, 47)
#    df1['pc-48-63'] = df['pc-top'] - df1['pc-32-47']

    # effective address
    df1['ea-00-15'] = df['effective_address'] % math.pow(2, 15)
    df['ea-top']    = df['effective_address'] - df1['ea-00-15']
    df1['ea-16-31'] = df['ea-top'] % math.pow(2, 31)
    df['ea-top']    = df['ea-top'] - df1['ea-16-31']
    df1['ea-32-47'] = df['ea-top'] % math.pow(2, 47)
    df1['ea-48-63'] = df['ea-top'] - df1['ea-32-47']

    # val0
    df1['v0-00-15'] = df['val0'] % math.pow(2, 15)
    df['v0-top']    = df['val0'] - df1['v0-00-15']
    df1['v0-16-31'] = df['v0-top'] % math.pow(2, 31)
    df['v0-top']    = df['v0-top'] - df1['v0-16-31']
    df1['v0-32-47'] = df['v0-top'] % math.pow(2, 31)
    df1['v0-48-63'] = df['v0-top'] - df1['v0-32-47']

    # Scale all values in df1 into the interval [0,1]
    cols = ['pc-00-15', 'pc-16-31', 'ea-00-15', 'ea-16-31', 'ea-32-47', \
            'ea-48-63', 'v0-00-15', 'v0-16-31', 'v0-32-47', 'v0-48-63']
    for col in cols:
        if df1[col].max() != 0:
            df1[col] = df1[col] / df1[col].max()            

    # Add columns for the previous k loads
    for i in range(1, k+1):
        df1['v0-00-15-' + str(i)] = df1['v0-00-15'].shift(i)
        df1['v0-16-31-' + str(i)] = df1['v0-16-31'].shift(i)
        df1['v0-32-47-' + str(i)] = df1['v0-32-47'].shift(i)
        df1['v0-48-63-' + str(i)] = df1['v0-48-63'].shift(i)

    # Add the store value's addresses, (split them up too)
    df1['sea-00-15-1']  = df['store-1ea'] % math.pow(2, 15)
    df['store-1ea-top'] = df['store-1ea'] - df1['sea-00-15-1']
    df1['sea-16-31-1']  = df['store-1ea-top'] % math.pow(2, 31)
    df['store-1ea-top'] = df['store-1ea-top'] - df1['sea-16-31-1']
    df1['sea-32-47-1']  = df['store-1ea-top'] % math.pow(2, 47)
    df1['sea-48-63-1']  = df['store-1ea-top'] - df1['sea-32-47-1']

    # previous store value
    df1['sv0-00-15-1']    = df['store-1val0'] % math.pow(2, 15)
    df['store-1val0-top'] = df['store-1val0'] - df1['sv0-00-15-1']
    df1['sv0-16-31-1']    = df['store-1val0-top'] % math.pow(2, 31)
    df['store-1val0-top'] = df['store-1val0-top'] - df1['sv0-16-31-1']
    df1['sv0-32-47-1']    = df['store-1val0-top'] % math.pow(2, 47)
    df1['sv0-48-63-1']    = df['store-1val0-top'] - df1['sv0-32-47-1']

    # the ea of the store before the previous store
    df1['sea-00-15-2']  = df['store-2ea'] % math.pow(2, 15)
    df['store-2ea-top'] = df['store-2ea'] - df1['sea-00-15-2']
    df1['sea-16-31-2']  = df['store-2ea-top'] % math.pow(2, 31)
    df['store-2ea-top'] = df['store-2ea-top'] - df1['sea-16-31-2']
    df1['sea-32-47-2']  = df['store-2ea-top'] % math.pow(2, 47)
    df1['sea-48-63-2']  = df['store-2ea-top'] - df1['sea-32-47-2']

    # the value of the store before the previous store
    df1['sv0-00-15-2']    = df['store-2val0'] % math.pow(2, 15)
    df['store-2val0-top'] = df['store-2val0'] - df1['sv0-00-15-2']
    df1['sv0-16-31-2']    = df['store-2val0-top'] % math.pow(2, 31)
    df['store-2val0-top'] = df['store-2val0-top'] - df1['sv0-16-31-2']
    df1['sv0-32-47-2']    = df['store-2val0-top'] % math.pow(2, 47)
    df1['sv0-48-63-2']    = df['store-2val0-top'] - df1['sv0-32-47-2']

    cols = ['sea-00-15-1', 'sea-16-31-1', 'sea-32-47-1', 'sea-48-63-1', \
            'sv0-00-15-1', 'sv0-16-31-1', 'sv0-32-47-1', 'sv0-48-63-1', \
            'sea-00-15-2', 'sea-16-31-2', 'sea-32-47-2', 'sea-48-63-2', \
            'sv0-00-15-2', 'sv0-16-31-2', 'sv0-32-47-2', 'sv0-48-63-2']

    # Scale all values in df1 into the interval [0,1]
    for col in cols:
        if df1[col].max() != 0:
            df1[col] = df1[col] / df1[col].max() 

    # Reorder the columns so the 'outputs' are the last 4 columns
    df1 = df1[[ 'pc-00-15', 'pc-16-31', 'ea-00-15', 'ea-16-31', 'ea-32-47', \
                'ea-48-63', 'v0-00-15-1', 'v0-16-31-1', 'v0-32-47-1', 'v0-48-63-1', \
                'v0-00-15-2', 'v0-16-31-2', 'v0-32-47-2', 'v0-48-63-2', \
                'v0-00-15-3', 'v0-16-31-3', 'v0-32-47-3', 'v0-48-63-3', \
                'v0-00-15-4', 'v0-16-31-4', 'v0-32-47-4', 'v0-48-63-4', \
                'sea-00-15-1', 'sea-16-31-1', 'sea-32-47-1', 'sea-48-63-1', \
                'sv0-00-15-1', 'sv0-16-31-1', 'sv0-32-47-1', 'sv0-48-63-1', \
                'sea-00-15-2', 'sea-16-31-2', 'sea-32-47-2', 'sea-48-63-2', \
                'sv0-00-15-2', 'sv0-16-31-2', 'sv0-32-47-2', 'sv0-48-63-2', \
                'v0-00-15', 'v0-16-31', 'v0-32-47', 'v0-48-63']]

    return df1.dropna()


def input_response_split(df):
    """ Split df into input output pairs. Columns assumed (in this order)
        pc-l, ea-l, ea-u, v0-l, v0-u, v0l-1, v0u-1, v0l-2, v0u-2, v0l-3, v0u-3, 
        v0l-4, v0u-4, s-1l, s-1u
        The output is given by v0-l, v0-u and all others form the inputs
    """
    data = df.values
    # Input  := all rows, all columns before the last 4
    # output := all rows, the last 4 columns
    return data[:, : -4], data[:, -4:]

def test_train_split(df, num_training_examples):
    """ Splits the dataframe into a test set and training set. Uses
        input_response_split to produce X, y then splits these into
        training and testing sets.
        usage: 
        X_tr, X_test, y_tr, y_test = test_train_split(df, num_training_examples)
    """
    assert(num_training_examples < len(df))
    X, y = input_response_split(df)
    X_tr = X[:num_training_examples,]
    X_te  = X[num_training_examples:,]
    y_tr = y[:num_training_examples]
    y_te  = y[num_training_examples:]
    return X_tr, X_te, y_tr, y_te

""" Models
"""
def build_model(do1=100, do2=100, do3=100, do4=100, do5=100, l_rate=0.00075):
    model = Sequential()
    model.add(Dense(do1,input_dim=38, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='sigmoid'))
    adam = Adam(learning_rate=l_rate)
    model.compile(optimizer=adam, loss='mse')
    return model

def get_rmse(pred, truth, n):
    rmse = 0
    for i in range(n):
        rmse += (truth[i][0] - pred[i][0])**2 + (truth[i][1] - pred[i][1])**2 + \
                (truth[i][2] - pred[i][2])**2 + (truth[i][3] - pred[i][3])**2
    return math.sqrt(rmse)/(2*n)