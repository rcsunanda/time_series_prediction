"""
TimeSeriesGenerator class
"""

import data_point as data_point

import scipy.stats as st
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler


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



###################################################################################################
"""
Generates a time series from given multivariable function
"""

def prepare_dataset(series_points, input_timesteps, output_timesteps):

    dataset_size = len(series_points)
    data_dim = len(series_points[0].X)

    # Cannot go beyond (not enough datapoints to gather past and future points)
    prepared_dataset_size = dataset_size - input_timesteps - output_timesteps

    X = np.zeros((prepared_dataset_size, input_timesteps, data_dim))
    Y = np.zeros((prepared_dataset_size, data_dim * output_timesteps))

    for idx in range(prepared_dataset_size):
        # Prepare X with current and previous timesteps
        step = 0
        for j in range(idx, idx + input_timesteps):
            point = series_points[j]
            for dim in range(data_dim):
                X[idx][step][dim] = point.X[dim]
            step += 1

        # Prepare Y with next timesteps
        future_start = idx + input_timesteps
        step = 0
        for k in range(future_start, future_start + output_timesteps):
            future_point = series_points[k]
            for dim in range(data_dim):
                Y[idx][step+dim] = future_point.X[dim]
            step += data_dim

    # X is numpy array with shape (dataset_size, input_timesteps, data_dim)
    # Y is numpy array with shape (dataset_size, data_dim)
    return (X, Y)


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

    assert first_ts_sample_set.shape[1] == input_timesteps
    assert data_dimension == 1  # Currently we only support this (otherwise need a lot of adjustments to prepare current-sample inside the loop)

    current_sample_set_for_prediction = first_ts_sample_set
    all_predictions = np.zeros((num_predictions * output_timesteps))

    for i in range (num_predictions):
        predicted_sample = model.predict(current_sample_set_for_prediction)
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
#
#
# def prepare_dataset(data_dim, series, time_steps):
#     dataset_size = len(series)
#
#     X = np.zeros((dataset_size, data_dim))
#     Y = np.zeros((dataset_size, data_dim))
#
#     # Append a copy last element to series (to prepare Y)
#     last_point = series[-1]
#     series.append(data_point.DataPoint(
#         last_point.t, last_point.X, last_point.true_is_anomaly, last_point.predicted_is_anomaly))
#
#     for idx in range(dataset_size):
#         curr_point = series[idx]
#         next_point = series[idx + 1]
#         for dim in range(data_dim):
#             X[idx][dim] = curr_point.X[dim]
#             Y[idx][dim] = next_point.X[dim]
#
#     X = X.reshape((dataset_size, time_steps, data_dim))
#     Y = Y.reshape((dataset_size, data_dim))  ## Is this necessary? Isn't Y already in this shape?
#
#     # X is numpy array with shape (dataset_size, time_steps, data_dim)
#     # Y is numpy array with shape (dataset_size, data_dim)
#     return (X, Y)


# series is a n-dimensional time series in a numpy ndarray format
# Return our standard time series type (list of DataPoints)
def convert_to_series(input_series, t_range):
    assert (len(t_range) == 2)
    # assert data_dimnsion == 1   # Don't support more for now

    dataset_size = input_series.shape[0]

    data_dim = input_series.shape[1]

    t_vals = np.linspace(t_range[0], t_range[1], num=dataset_size)

    output_series = []

    for idx in range(dataset_size):
        sample_X = []
        t = t_vals[idx]

        for dim in range(data_dim): # This is also wrong, without considering output_timesteps
            dim_sample_val = input_series[idx][dim]
            sample_X.append(dim_sample_val)

        output_series.append(data_point.DataPoint(t, sample_X, -1, -1))

    return output_series


def new_convert_to_series(input_series, t_range):
    assert (len(t_range) == 2)
    # assert data_dimnsion == 1   # Don't support more for now

    dataset_size = input_series.shape[0]

    t_vals = np.linspace(t_range[0], t_range[1], num=dataset_size)

    output_series = []

    for idx in range(dataset_size):
        sample_X = [input_series[idx]]
        t = t_vals[idx]
        output_series.append(data_point.DataPoint(t, sample_X, -1, -1))

    return output_series



class TimeSeriesGenerator:
    def __init__(self, num_dimensions, functions):
        self.num_dimensions = num_dimensions
        self.set_functions(functions)


    def __repr__(self):
        return "TimeSeriesGenerator(\n\tnum_dimensions={} \n\tfunctions={} \n)"\
            .format(self.num_dimensions, self.functions)


    def set_functions(self, functions):
        assert (len(functions) == self.num_dimensions)
        self.functions = functions  # Function for each dimension used for generating time series


    # Generate time series of given size
    def generate_time_series(self, t_range, count, is_anomolous):
        assert (len(t_range) == 2)
        t_vals = np.linspace(t_range[0], t_range[1], num=count)
        series = []
        # Sample value for each dimension
        for idx in range(count):
            sample_X = []
            t = t_vals[idx]

            for dim in range(self.num_dimensions):
                func = self.functions[dim]
                dim_sample_val = func(t)
                sample_X.append(dim_sample_val)

            series.append(data_point.DataPoint(t, sample_X, is_anomolous, -1))

        return series


def calculate_errors(actual_series, forecast_series):
    num_points = min(len(actual_series), len(forecast_series))

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






