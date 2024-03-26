# taken from: https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
#%%
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
# create sequence
length = 270*20*15 + 1
sequence = [i/float(length) for i in range(length)]
# create X/y pairs; X : input, y:output
df = DataFrame(sequence)
df = concat([df, df.shift(1)], axis=1)
df.dropna(inplace=True)
# convert to LSTM friendly format
values = df.values
X, y = values[:, 0], values[:, 1]
X = X.reshape(270,20,15)
y = y.reshape(270,20,15)
# configure network
n_batch = 1
n_epoch = 22
n_neurons = 10
# design network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
#%%

history = model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=False)

# %%
# plot results
plt.plot(history.history['loss'])
plt.legend()

# %%
