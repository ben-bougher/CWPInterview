import pandas as pd
from datetime import timedelta
from sklearn import preprocessing
import numpy as np


import keras
from keras.models import Model, Sequential
from keras.layers import Dense, LSTM, Input, Activation

from keras.wrappers.scikit_learn import KerasRegressor

epochs = 25
batch_size = 25


def parse_hub_data(xls_file, hub_name, weather_station, load_station):

    df = pd.read_excel(xls_file, sheetname=[0, 1, 2])

    market_data = df[0]
    weather_data = df[1]
    load_data = df[2]

    hub_df = market_data.loc[market_data['nodename'] ==
                             hub_name][["localday", "he", "rtlmp", "dalmp"]]
    hub_df.rename(columns={'he': 'hour', 'localday': 'datetime'}, inplace=True)

    hub_df["hour"] -= 1
    hub_df["datetime"] += [timedelta(hours=int(h)) for h in hub_df.hour]

    min_date = hub_df["datetime"].min()
    max_date = hub_df["datetime"].max()

    weather_df = \
        weather_data.loc[weather_data.station_name == weather_station][[
            "value", "datetime"]]
    weather_df = weather_df.loc[(weather_data.datetime >= min_date) & (
        weather_data.datetime <= max_date)]
    weather_df.rename(columns={"value": "temperature"}, inplace=True)

    load_df = load_data.loc[(load_data.localdatetime >= min_date) &
                            (load_data.localdatetime <= max_date)][[
                                load_station + "_actual_load", load_station +
                                "_mtlf", "localdatetime"]]
    load_df.rename(columns={"localdatetime": "datetime",
                            load_station + "_actual_load": "actual_load",
                            load_station + "_mtlf": "mtlf"}, inplace=True)

    weather_df = weather_df.set_index("datetime")
    hub_df = hub_df.set_index("datetime")
    load_df = load_df.set_index("datetime")

    hub_df = hub_df.join([weather_df, load_df]).dropna()
    hub_df = hub_df[~hub_df.index.duplicated(keep='first')]

    return hub_df


def standardize_data(df):

    data = df.values
    transform = preprocessing.MinMaxScaler(feature_range=[-1, 1]).fit(data)
    scaled = transform.transform(data)

    return scaled, transform


# def extract_daily_profile(df):
#
#     frame = pd.DataFrame(index=df.index, columns=range(24))
#
#     for i, date in enumerate(frame.index[7294:]):
#
#         start = date - np.timedelta64(2, 'D')
#         end = date - np.timedelta64(1, 'D')
#
#         values = df[start:date]["rtlmp"].values[:24]
#         if values.size != 24:
#             continue
#
#         frame.loc[date] = values
#
#     return frame


def extract_lagged_series(df, lags):

    output = pd.DataFrame(index=df.index)

    for lag in lags:
        output["lag_" + str(lag)] = df["rtlmp"].shift(lag)
    return output


def extract_day_of_week(df):

    return [d.isoweekday() for d in df.index]


def extract_demand_gradient(df):

    data = df["mtlf"]
    output = data - np.roll(data.values, 1)

    output[0] = np.nan

    return output


def extract_exogenous_variables(df):

    output = pd.DataFrame(index=df.index)
    output["forecast_temperature"] = df["temperature"]
    output["forecast_load"] = df["mtlf"]
    output["hour"] = df["hour"]
    output["day_of_week"] = extract_day_of_week(df)
    output["change_in_demand_prediction"] = extract_demand_gradient(df)

    return output


def clean_data(timeseries, exo, labels):

    d = exo.join(timeseries)
    d["labels"] = labels

    d = d.dropna()

    return d[timeseries.columns], d[exo.columns], d["labels"]


def get_standardized_matrices(ts, exo, labels, split=8000, test_size=1000):

    ts_standard, ts_transform = standardize_data(ts)
    exo_standard, exo_transform = standardize_data(exo)
    labels_standard, labels_transform = standardize_data(labels)

    return (np.expand_dims(ts_standard[:split], axis=1),
            np.expand_dims(ts_standard[split:split + test_size], axis=1),
            exo_standard[:split],
            exo_standard[split: split + test_size], labels_standard[:split],
            labels_standard[split: split + test_size],
            ts_transform, exo_transform, labels_transform)


def build_RNN(n_exogenous_features):

    tsteps = 1

    ts_input = Input(shape=(1, 1), dtype='float32', name='ts_input',
                     batch_shape=(batch_size, 1, 1))

    lstm_out = LSTM(150, input_shape=(tsteps, 1), batch_size=batch_size,
                    return_sequences=False,
                    stateful=True)(ts_input)

    lstm_predict = Dense(1, activation='linear', name='lstm_predict')(lstm_out)

    exo_input = Input(shape=(1, n_exogenous_features), dtype='float32',
                      name='exo_input', batch_shape=(batch_size,
                                                     n_exogenous_features))
    x = keras.layers.concatenate([lstm_out, exo_input])
    x = Dense(5, activation='tanh', batch_size=batch_size)(x)
    x = Dense(5, activation='tanh', batch_size=batch_size)(x)
    main_output = Dense(1, activation='linear', name='main_output',
                        batch_size=batch_size)(x)

    model = Model(inputs=[ts_input, exo_input],
                  outputs=[lstm_predict, main_output])
    model.compile(loss="mean_squared_error", optimizer='adam',
                  loss_weights=[1, 5])

    return model


def train_RNN(model, ts_train, ex_train, truth_train):

    print('Training')
    for i in range(epochs):

        model.fit([ts_train, ex_train],
                  [np.reshape(truth_train, (truth_train.shape[0], 1)),
                   truth_train],
                  batch_size=batch_size,
                  epochs=1,
                  verbose=1,
                  shuffle=False)
        model.reset_states()

    return model


def build_ANN(n_exogenous_features, n_time_lags):

    def build():
        model = Sequential()

        model.add(Dense(units=10,
                        input_dim=n_exogenous_features + n_time_lags))

        model.add(Activation('tanh'))
        model.add(Dense(units=5))
        model.add(Activation('tanh'))
        model.add(Dense(units=1, activation="linear"))

        model.compile(loss='mean_squared_error', optimizer='adam')

        return model

    # fix random seed for reproducibility
    seed = 90
    np.random.seed(seed)

    # evaluate model with standardized dataset
    estimator = KerasRegressor(
        build_fn=build, batch_size=100, nb_epoch=5, epochs=20)

    return estimator
