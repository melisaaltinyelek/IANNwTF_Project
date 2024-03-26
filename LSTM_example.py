# taken from: https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
#%%
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt

# sample dataset
# create sequence
length = 270*20*15
length2 = 270*20*9
sequence = [i/float(length) for i in range(length)]
sequence2 = [i/float(length2) for i in range(length2)]
# create X/y pairs; X : input, y:output
X = np.array(sequence).reshape(270,20,15)
y = np.array(sequence2).reshape(270,20,9)
# configure network
n_batch = 1
n_epoch = 22
n_neurons = 10
#%%
# design and fit the network
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
model.add(Dense(9))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=False)
# %%
# plot results
plt.plot(history.history['loss'])
plt.legend()

# %%
