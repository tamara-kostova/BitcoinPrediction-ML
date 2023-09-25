from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Dropout
import plotly.offline as py
import plotly.graph_objs as graph
from sklearn.preprocessing import MinMaxScaler

# Prepare Data
def prepare_data(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


dataset = pd.read_csv(filepath_or_buffer="input/bitcoininputdata", index_col="Date")
dataset['Weighted Price'].replace(0, np.nan, inplace=True)
dataset['Weighted Price'].fillna(method='ffill', inplace=True)

btc = graph.Scatter(x=dataset.index, y=dataset['Weighted Price'], name= 'Price')
py.plot([btc])

values = dataset['Weighted Price'].values.reshape(-1,1)
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
values= scaler.fit_transform(values)

train_size = int(len(values) * 0.7)
test_size = len(values) - train_size
train, test = values[0:train_size, :], values[train_size:len(values), :]

look_back = 1
train_x, train_y = prepare_data(train, look_back)
test_x, test_y = prepare_data(test, look_back)
train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# Model - LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape=(train_x.shape[1], train_x.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mae', optimizer='adam')
metadata = model_lstm.fit(train_x, train_y, epochs=300, batch_size=100, validation_data=(test_x, test_y), verbose=0, shuffle=False)

pyplot.plot(metadata.history['loss'], label='train')
pyplot.plot(metadata.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model_lstm.predict(test_x)
pyplot.plot(yhat, label='predicted')
pyplot.plot(test_y, label='true')
pyplot.legend()
pyplot.show()

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_y_inverse = scaler.inverse_transform(test_y.reshape(-1, 1))
lstm_rmse = sqrt(mean_squared_error(test_y_inverse, yhat_inverse))
print('LSTM RMSE: %.3f' % lstm_rmse)

pyplot.plot(yhat_inverse, label='predicted')
pyplot.plot(test_y_inverse, label='actual', alpha=0.5)
pyplot.legend()
pyplot.show()

dates = dataset.tail(len(test_x)).index
test_y_reshape = test_y_inverse.reshape(len(test_y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
actual_chart = graph.Scatter(x=dates, y=test_y_reshape, name= 'Actual Price')
predict_chart = graph.Scatter(x=dates, y=yhat_reshape, name= 'Predict Price')
py.plot([predict_chart, actual_chart])

# Model - GRU
model_gru = Sequential()
model_gru.add(GRU(100, input_shape=(train_x.shape[1], train_x.shape[2])))
model_gru.add(Dropout(0.2))
model_gru.add(Dense(units=1))
model_gru.compile(loss='mae', optimizer='adam')
metadata = model_gru.fit(train_x, train_y, epochs=300, batch_size=100, validation_data=(test_x, test_y), verbose=0, shuffle=False)

pyplot.plot(metadata.history['loss'], label='train')
pyplot.plot(metadata.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

yhat = model_gru.predict(test_x)
pyplot.plot(yhat, label='predicted')
pyplot.plot(test_y, label='true')
pyplot.legend()
pyplot.show()

yhat_inverse = scaler.inverse_transform(yhat.reshape(-1, 1))
test_y_inverse = scaler.inverse_transform(test_y.reshape(-1, 1))
gru_rmse = sqrt(mean_squared_error(test_y_inverse, yhat_inverse))
print('GRU RMSE: %.3f' % gru_rmse)

pyplot.plot(yhat_inverse, label='predicted')
pyplot.plot(test_y_inverse, label='actual', alpha=0.5)
pyplot.legend()
pyplot.show()

dates= dataset.tail(len(test_x)).index
test_y_reshape = test_y_inverse.reshape(len(test_y_inverse))
yhat_reshape = yhat_inverse.reshape(len(yhat_inverse))
actual_chart = graph.Scatter(x=dates, y=test_y_reshape, name= 'Actual Price')
predict_chart = graph.Scatter(x=dates, y=yhat_reshape, name= 'Predict Price')
py.plot([predict_chart, actual_chart])


# Prepare Data for Supervised Learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    ready = pd.concat(cols, axis=1)
    ready.columns = names
    if dropnan:
        ready.dropna(inplace=True)
    return ready


values = dataset[['Weighted Price'] + ['Volume (BTC)'] + ['Volume (Currency)']].values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

reframed = series_to_supervised(values, 1, 1)
reframed.head()
reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
n_train_hours = int(len(values) * 0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# Model - LSTM for Supervised Learning
multi_model = Sequential()
multi_model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
multi_model.add(Dense(1))
multi_model.compile(loss='mae', optimizer='adam')
metadata = multi_model.fit(train_X, train_y, epochs=300, batch_size=100, validation_data=(test_X, test_y), verbose=0, shuffle=False)

pyplot.plot(metadata.history['loss'], label='multi_train')
pyplot.plot(metadata.history['val_loss'], label='multi_test')
pyplot.legend()
pyplot.show()

yhat = multi_model.predict(test_X)
pyplot.plot(yhat, label='predicted')
pyplot.plot(test_y, label='true')
pyplot.legend()
pyplot.show()

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]

supervised_rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Supervised RMSE: %.3f' % supervised_rmse)

actual_chart = graph.Scatter(x=dates, y=inv_y, name= 'Actual Price')
multi_predict_chart = graph.Scatter(x=dates, y=inv_yhat, name= 'Multi Predict Price')
predict_chart = graph.Scatter(x=dates, y=yhat_reshape, name= 'Predict Price')
py.plot([predict_chart, multi_predict_chart, actual_chart])
