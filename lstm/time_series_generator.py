"""
TimeSeriesGenerator class
"""

import data_point as data_point

import scipy.stats as st
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from keras.utils import to_categorical


def scale_series(series, scaler=None):

    dataset_size = len(series)
    data_dim = len(series[0].X)

    X = np.zeros((dataset_size, data_dim))

    for i, point in enumerate(series):
        X[i] = point.X

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X)

    scaled = scaler.transform(X)

    for i, point in enumerate(series):
        point.X = scaled[i]

    return scaler


def descale_series(series, scaler):

    dataset_size = len(series)
    data_dim = len(series[0].X)

    X = np.zeros((dataset_size, data_dim))

    for i, point in enumerate(series):
        X[i] = point.X

    X = scaler.inverse_transform(X)

    for i, point in enumerate(series):
        point.X = X[i]


def engineer_features_from_time(series_points):
    num_points = len(series_points)
    months = np.zeros(num_points)

    for i, point in enumerate( series_points):
        # Parse timestamp and compute necessary values
        date_obj = datetime.strptime(point.timestamp, '%Y-%m-%d')
        month = date_obj.month - 1  # 0 - 11
        quarter = (month) // 3  # 0 - 3
        day_of_week = date_obj.weekday()    # 0 - 6
        day_of_month = date_obj.day # 1- 31

        is_weekend = 0
        if day_of_week == 5 or day_of_week == 6:
            is_weekend = 1

        month_encoded = to_categorical(month, num_classes=12)
        quarter_encoded = to_categorical(quarter, num_classes=4)
        day_of_week_encoded = to_categorical(quarter, num_classes=6)

        new_feature_vec = np.copy(point.X)
        new_feature_vec = np.concatenate((new_feature_vec, month_encoded))
        new_feature_vec = np.concatenate((new_feature_vec, quarter_encoded))
        new_feature_vec = np.concatenate((new_feature_vec, day_of_week_encoded))
        new_feature_vec = np.concatenate((new_feature_vec, [is_weekend]))
        new_feature_vec = np.concatenate((new_feature_vec, [day_of_month]))

        dimension = len(new_feature_vec)

        point.X = new_feature_vec

        # print('{} - month={}, quarter={}, day_of_month={}, day_of_week={}, is_weekend={}'.
        #       format(date_obj, month, quarter, day_of_month, day_of_week, is_weekend))

        # print('month_encoded={}'.format(month_encoded))
        # print(new_feature_vec)

    return dimension


###################################################################################################
def prepare_dataset(series_points, input_timesteps, predict_dimensions, output_timesteps):

    num_predict_dimensions = len(predict_dimensions)

    assert output_timesteps == 1
    assert num_predict_dimensions == 1

    predict_dimension = predict_dimensions[0]
    dataset_size = len(series_points)
    input_dimension = len(series_points[0].X)

    # Cannot go beyond (not enough datapoints to gather past and future points)
    prepared_dataset_size = dataset_size - input_timesteps - output_timesteps

    X = np.zeros((prepared_dataset_size, input_timesteps, input_dimension))
    Y = np.zeros((prepared_dataset_size, 1))

    for idx in range(prepared_dataset_size):
        # Prepare X with current and previous timesteps
        step = 0
        for j in range(idx, idx + input_timesteps):
            point = series_points[j]
            for dim in range(input_dimension):
                X[idx][step][dim] = point.X[dim]
            step += 1

        next_point = series_points[idx + input_timesteps]
        Y[idx][0] = next_point.X[predict_dimension]

    # X is numpy array with shape (dataset_size, input_timesteps, input_dimension)
    # Y is numpy array with shape (dataset_size, 1)
    return (X, Y)


def snowballing_predict(model, initial_ts_sample_set, X_to_pred, input_timesteps, output_timesteps):
    """
        Make predictions using given LSTM model by accumulating its own predictions, and using those to predict on next value etc
        initial_ts_sample_set = sample to use as the starting point for predictions (usually the last known sample of the time-series)
                            [this must be an np.ndarray with shape (num_samples, input_timesteps, dimension)]
        X_to_pred = a series of datapoints which contains the X features (engineered features) for to-be-predicted points
        input_timesteps = no. of input time steps used when training the LSTM model (input layer size)
        output_timesteps = no. of output time steps used when training the LSTM model (output layer size)
        return:
    """

    prediction_batch_size = 365    # Size of batch before LSTM state is reset (larger, the better, but may consume memory)

    num_samples = initial_ts_sample_set.shape[0]
    data_dimension = initial_ts_sample_set.shape[2]
    num_predictions = len(X_to_pred)

    assert output_timesteps == 1
    assert initial_ts_sample_set.shape[1] == input_timesteps

    current_input_set = initial_ts_sample_set   # This variable holds the input to be fed to model.predict(). It gets appended in each iteration of the for loop

    all_predictions = np.zeros(num_predictions)

    for i in range(num_predictions):
        predicted_set = model.predict(current_input_set, batch_size=prediction_batch_size)
        assert num_samples == predicted_set.shape[0]    # just to verify

        last_prediction = predicted_set[-1]
        all_predictions[i] = last_prediction

        # print("before - {}".format(X_to_pred[i].X[:]))

        new_datapoint = np.zeros(data_dimension)
        new_datapoint[:] = X_to_pred[i].X[:]
        new_datapoint[0] = last_prediction
        new_datapoint = new_datapoint.reshape([1, data_dimension])  # To make it possible to concat with the new_sample below

        # print("after - {}".format(X_to_pred[i].X[:]))

        new_sample = np.copy(current_input_set[-1]) # creating a new_sample based on the last one in the current set
        assert new_sample.shape[0] == input_timesteps
        assert new_sample.shape[1] == data_dimension
        new_sample = np.delete(new_sample, [0], axis=0)  # remove first element

        new_sample = np.concatenate((new_sample, new_datapoint), axis=0)
        new_sample = new_sample.reshape([1, input_timesteps, data_dimension])
        current_input_set = np.concatenate((current_input_set, new_sample), axis=0)

        num_samples += 1

    return all_predictions



def moving_forward_window_predict(model, first_ts_sample_set, input_timesteps, output_timesteps, num_predictions):
    """
    Make predictions using given LSTM model based on a window that moves forwards on its own predictions
    ts_last_sample = sample to use as the starting point for predictions (usually the last known sample of the time-series)
                        [this must be an np.ndarray with shape (num_samples, input_timesteps, dimension)]
    input_timesteps = no. of input time steps used when training the LSTM model (input layer size)
    output_timesteps = no. of output time steps used when training the LSTM model (output layer size)
    return:
    """
    num_samples = first_ts_sample_set.shape[0]
    data_dimension = first_ts_sample_set.shape[2]

    assert output_timesteps == 1
    assert first_ts_sample_set.shape[1] == input_timesteps
    assert data_dimension == 1  # Currently we only support this (otherwise need a lot of adjustments to prepare current-sample inside the loop)

    current_sample_set_for_prediction = first_ts_sample_set
    all_predictions = np.zeros((num_predictions * output_timesteps))

    for i in range (num_predictions):
        predicted_sample = model.predict(current_sample_set_for_prediction, batch_size=300)
        predicted_sample = predicted_sample[-1, :]  # we only need the last output_steps
        f1 = current_sample_set_for_prediction.flatten()
        f2 = predicted_sample.flatten()
        all_points = np.concatenate((f1, f2), axis=0)
        current_window = all_points[-input_timesteps*num_samples:]
        current_sample_set_for_prediction = current_window.reshape([num_samples, input_timesteps, data_dimension])

        start = i * output_timesteps
        end = start + output_timesteps
        all_predictions[start:end] = predicted_sample


    return all_predictions


def new_convert_to_series(input_series, t_range):
    assert (len(t_range) == 2)
    # assert data_dimnsion == 1   # Don't support more for now

    dataset_size = input_series.shape[0]

    t_vals = np.linspace(t_range[0], t_range[1], num=dataset_size)

    output_series = []

    for idx in range(dataset_size):
        sample_X = [input_series[idx]]
        t = t_vals[idx]
        output_series.append(data_point.DataPoint(t, '0000-00-00', sample_X, -1, -1))

    return output_series


def calculate_errors(actual_series, forecast_series):
    # print('len(actual_series)={}, len(forecast_series)={}'.format(len(actual_series), len(forecast_series)))
    # num_points = min(len(actual_series), len(forecast_series))

    assert len(actual_series) == len(forecast_series)
    num_points = len(actual_series)

    squared_error_sum = 0
    absolute_error_sum = 0
    absolute_percentage_error_sum = 0

    for i in range (num_points):
        actual_val = actual_series[i].X[0]  # first dim only
        forecast_val = forecast_series[i].X[0]  # first dim only

        diff = actual_val - forecast_val

        absolute_error_sum += abs(diff)
        squared_error_sum += diff ** 2

        if actual_val != 0:
            absolute_percentage_error_sum += abs(diff / actual_val)

    rmse = (squared_error_sum / num_points) ** (0.5)
    mae = absolute_error_sum / num_points
    mape = absolute_percentage_error_sum/ num_points

    return (rmse, mae, mape)






