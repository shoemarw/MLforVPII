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
    df['store-1val0'] = df['store-1val0'].astype('float64')
    df['store-2val0'] = df['store-2val0'].astype('float64')

    # Build a dataframe with the 64bit values split into 32 bit values
    df1 = pd.DataFrame(index=df.index)
    df1['pc-l'] = df['pc'] % math.pow(2, 31) # lower 32 bits
    df1['ea-l'] = df['effective_address'] % math.pow(2, 31)
    df1['ea-u'] = (df['effective_address'] - df1['ea-l'])/math.pow(2, 31) #remove trailing zeros
    df1['v0-l'] = df['val0'] % math.pow(2, 31)
    df1['v0-u'] = (df['val0'] - df1['v0-l'])/math.pow(2, 31) #remove trailing zeros

    # Scale all values in df1 into the interval [0,1]
    df1['pc-l'] = df1['pc-l']/df1['pc-l'].max()
    df1['ea-l'] = df1['ea-l']/df1['ea-l'].max()
    df1['ea-u'] = df1['ea-u']/df1['ea-u'].max()
    df1['v0-l'] = df1['v0-l']/df1['v0-l'].max()
    df1['v0-u'] = df1['v0-u']/df1['v0-u'].max()

    # Add columns for the previous k loads
    for i in range(1, k+1):
        lname = 'v0l-' + str(i)
        df1[lname] = df1['v0-l'].shift(i)
        uname = 'v0u-' + str(i)
        df1[uname] = df1['v0-u'].shift(i)

    # Add columns for the previous store values
    df1['s-1l'] = df['store-1val0'] % math.pow(2, 31) # lower 32 bits
    df1['s-1u'] = (df['store-1val0'] - df1['s-1l']) / math.pow(2, 31)
    df1['s-2l'] = df['store-2val0'] % math.pow(2, 31) # lower 32 bits
    df1['s-2u'] = (df['store-2val0'] - df1['s-2l']) / math.pow(2, 31)

    df1['s-1l'] = df1['s-1l'] / df1['s-1l'].max()
    df1['s-1u'] = df1['s-1u'] / df1['s-1u'].max()
    df1['s-2l'] = df1['s-2l'] / df1['s-2l'].max()
    df1['s-2u'] = df1['s-2u'] / df1['s-2u'].max()

    # Reorder the columns so the 'outputs' are the last 2 columns
    df1 = df1[["pc-l", "ea-l", "ea-u", "v0l-1", "v0u-1", "v0l-2", "v0u-2", "v0l-3", \
               "v0u-3", "v0l-4", "v0u-4", "s-1l", "s-1u", "s-2l", "s-2u", "v0-l", "v0-u"]]

    return df1.dropna()


def input_response_split(df):
    """ Split df into input output pairs. Columns assumed (in this order)
        pc-l, ea-l, ea-u, v0-l, v0-u, v0l-1, v0u-1, v0l-2, v0u-2, v0l-3, v0u-3, 
        v0l-4, v0u-4, s-1l, s-1u
        The output is given by v0-l, v0-u and all others form the inputs
    """
    data = df.values
    # Input  := all rows, all columns before the last 2
    # output := all rows, the last two columns
    return data[:, : -2], data[:, -2:]

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
    model.add(Dense(do1,input_dim=15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(do5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal', activation='sigmoid'))
    adam = Adam(learning_rate=l_rate)
    model.compile(optimizer=adam, loss='mse')
    return model

""" Evaluation
"""

def get_rmse(pred, truth, n):
    rmse = 0
    for i in range(n):
        rmse += (truth[i][0] - pred[i][0])**2 + (truth[i][1] - pred[i][1])**2
    return math.sqrt(rmse)/(2*n)