import time_series_generator as gen
import data_point as dp

import data_synthesizer
import matplotlib.pyplot as plt
import csv
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils.vis_utils import plot_model
from keras import regularizers

def load_series_from_file(start_timestamp, filename):
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        series = []
        time_index = start_timestamp    # Start here
        for i, row in enumerate(reader):
            if i > 0:   # Ignore first row with column names
                timestamp = row[0]
                sample_X = np.array(row[1:])    # Without the timestamp
                series.append(dp.DataPoint(time_index, timestamp, sample_X, False, False))
                time_index = time_index + 1/365

    return series


def get_data(start_timestamp, filename):
    print("get_data: Loading data from filename={}".format(filename))
    series = load_series_from_file(start_timestamp, filename)
    dimension = len(series[0].X)
    print("\t dimension={}".format(dimension))
    return dimension, series


def get_generated_data(t_range):
    dimension = 3
    series = data_synthesizer.generate_complex_series(dimension, t_range, count=1000, anomaly_rate=0)
    return dimension, series


def plot_series(series, title):
    t = [point.t for point in series]
    x1 = [point.X[0] for point in series]
    # x2 = [point.X[1] for point in series]

    plt.plot(t, x1, label=title, linewidth=0.5)
    # plt.plot(t, x2, label=title+'_dimension-2')

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend(loc='upper right')


###################################################################################################
"""
Fit an LSTM network to a 2-D time series prediction
"""


def plot_training_history(history):
    plt.figure()
    plt.title("Training history")
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')


def run_system():
    # Training and validation data
    dimension, train_series = get_data(2013, "data/item1_train.csv")
    # dimension, train_series = get_generated_data(t_range=(-1, 1))
    scaler = gen.scale_series(train_series)
    dimension = gen.engineer_features_from_time(train_series)
    # plot_series(train_series, "Training series")

    # dim_val, validation_series = get_generated_data(t_range=(1, 1.5))
    dim_val, validation_series = get_data(2016, "data/item1_validation.csv")
    gen.scale_series(validation_series)
    gen.engineer_features_from_time(validation_series)

    # LSTM Architecture

    input_timesteps = 10
    output_timesteps = 1  # This must be 1, because for now we only have the capability to predict one-step ahead
    predict_dimensions = [0]    # Which dimensions in the input series to predict. There must be only one dimension for now
    num_predict_dimensions = len(predict_dimensions)

    assert output_timesteps == 1
    assert num_predict_dimensions == 1

    input_layer_units = dimension   # Dimensionality of time series data
    hidden_layer_1_units = 10
    hidden_layer_2_units = 5
    hidden_layer_3_units = 10
    hidden_layer_4_units = 10
    output_layer_units = num_predict_dimensions * output_timesteps


    # Training params
    batch_size = 25 # Mini batch size in GD/ other algorithm
    epcohs = 100 # 50 is good

    loss_function = 'mse'
    activation_function = 'tanh'
    learning_rate = 0.001
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    # optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=0.1)
    l2_regularization = 0.1
    dropout_rate = 0.35

    print("dimension={}, input_timesteps={}, output_timesteps={}, predict_dimensions={}, input_layer_units={}, "
          "batch_size={}, epochs={}, loss_function={}, optimizer={}".
          format(dimension, input_timesteps, output_timesteps, predict_dimensions, input_layer_units,
                 batch_size, epcohs, loss_function, optimizer))


    # Create network
    model = Sequential()

    # model.add(LSTM(hidden_layer_1_units, batch_input_shape=(batch_size, input_timesteps, input_layer_units), stateful=True))  # Does not work
    model.add(LSTM(hidden_layer_1_units, activation=activation_function, return_sequences=True, input_shape=(input_timesteps, input_layer_units)))
    # model.add(Dropout(rate=dropout_rate))
    model.add(LSTM(hidden_layer_2_units, activation=activation_function, return_sequences=False))
    # model.add(Dropout(rate=dropout_rate))
    # model.add(LSTM(hidden_layer_3_units, activation=activation_function, return_sequences=True))
    # model.add(LSTM(hidden_layer_4_units, activation=activation_function, return_sequences=False))
    model.add(Dense(output_layer_units))

    model.compile(loss=loss_function, optimizer=optimizer)

    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    print("Model diagram was written to: model_plot.png")


    # Train network
    X_train, Y_train = gen.prepare_dataset(train_series, input_timesteps, predict_dimensions, output_timesteps)
    X_validation, Y_validation = gen.prepare_dataset(validation_series, input_timesteps, predict_dimensions, output_timesteps)

    print("Training LSTM model")
    history = model.fit(X_train, Y_train, epochs=epcohs, batch_size=batch_size, verbose=2
                        , validation_data=(X_validation, Y_validation))
    print("Training LSTM model complete")

    plot_training_history(history)

    print("Predicting on validation data")
    Y_validation_predicted = gen.snowballing_predict(model, X_train[-100:], validation_series, input_timesteps, output_timesteps)


    validation_t_range = (validation_series[0].t, validation_series[-1].t)

    Y_validation_series = gen.new_convert_to_series(Y_validation, validation_t_range)
    Y_validation_predicted_series = gen.new_convert_to_series(Y_validation_predicted, validation_t_range)


    # Predict on test data
    dim2, test_series = get_data(2017, "data/item1_test.csv")
    # dim2, test_series = get_generated_data(t_range=(1.5, 2.5))
    gen.scale_series(test_series, scaler)
    gen.engineer_features_from_time(test_series)
    test_t_range = (test_series[0].t, test_series[-1].t)
    X_test, Y_test = gen.prepare_dataset(test_series, input_timesteps, predict_dimensions, output_timesteps)

    print("Predicting on test data")
    Y_test_predicted = gen.snowballing_predict(model, X_validation[-100:], test_series, input_timesteps, output_timesteps)

    Y_test_predicted_series = gen.new_convert_to_series(Y_test_predicted, test_t_range)
    Y_test_series = gen.new_convert_to_series(Y_test, test_t_range)


    # Report error metrics
    val_rmse, val_mae, val_mape = gen.calculate_errors(validation_series, Y_validation_predicted_series)
    test_rmse, test_mae, test_mape = gen.calculate_errors(test_series, Y_test_predicted_series)

    print("Validation errors:\n RMSE=\n{}\n MAE=\n{}\n MAPE=\n{}".format(val_rmse, val_mae, val_mape))
    print("Test errors:\n RMSE=\n{}\n MAE=\n{}\n MAPE=\n{}".format(test_rmse, test_mae, test_mape))


    # # Predict on training data
    # Y_predicted_on_train = model.predict(X_train)
    # train_t_range = (train_series[0].t, train_series[-1].t)
    # Y_predicted_on_train_series = gen.new_convert_to_series(Y_predicted_on_train, train_t_range)

    # Plot training data predictions
    # plt.figure()
    # plot_series(train_series, "Y_train_set_true")
    # plot_series(Y_predicted_on_train_series, "Y_predicted_on_train_set")
    # plt.title("Train set prediction")


    # Plot validation and test data predictions

    gen.descale_series(train_series, scaler)
    # gen.descale_series(Y_validation_series, scaler)
    gen.descale_series(Y_validation_predicted_series, scaler)
    gen.descale_series(validation_series, scaler)
    gen.descale_series(test_series, scaler)
    # gen.descale_series(Y_test_series, scaler)
    gen.descale_series(Y_test_predicted_series, scaler)

    val_rmse, val_mae, val_mape = gen.calculate_errors(validation_series, Y_validation_predicted_series)
    test_rmse, test_mae, test_mape = gen.calculate_errors(test_series, Y_test_predicted_series)

    print("Validation errors:\n RMSE=\n{}\n MAE=\n{}\n MAPE=\n{}".format(val_rmse, val_mae, val_mape))
    print("Test errors:\n RMSE=\n{}\n MAE=\n{}\n MAPE=\n{}".format(test_rmse, test_mae, test_mape))


    # Concat validation and test series
    validation_test_series = validation_series + test_series
    validation_test_predicted_series = Y_validation_predicted_series + Y_test_predicted_series

    plt.figure()
    plot_series(train_series, "Training_series")
    plot_series(validation_test_series, "Validation+Test_true_series")
    plot_series(validation_test_predicted_series, "Validation+Test_forecast_series")
    # plt.title("LSTM forecasting of the function exp(x/2) * sin(2 * pi * x)")
    plt.title("LSTM forecasting of daily sales values")


    plt.show()


###################################################################################################

# Call test functions

run_system()
