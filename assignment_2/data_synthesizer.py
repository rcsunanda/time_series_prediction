
import data_point
import numpy as np
import math

# t_range is a tuple with (start, end) for time variable
def generate_trig_series(dim, t_range, count, anomaly_rate=0):
    def dim1_func(x, d):
        return math.sin(x)

    def dim2_func(x, d):
        return math.cos(x)

    def dim3_func(x, d):
        theta = math.pi / 6
        return 3*math.sin(x+theta)

    if dim == 1:
        functions = [dim1_func]
    elif dim == 2:
        functions = [dim1_func, dim2_func]
    elif dim == 3:
        functions = [dim1_func, dim2_func, dim3_func]
    else:
        assert False

    series = generate_time_series(dim=dim, t_range=t_range, count=count,
                                      functions=functions, anomaly_rate=anomaly_rate,
                                      add_noise=True, noise_var=0.01)

    return series


def generate_complex_series(dim, t_range, count, anomaly_rate=0):

    def damped_sine(x, d):
        return (math.e**(x*0.5)) * (math.sin(2 * math.pi * x))

    def cubic_func(x, d):
        return x**3 - 6 * x**2 + 4*x + 12

    def freq_increasing_sine(x, d):
        return math.sin(2 * math.pi * math.e**x)

    if dim == 1:
        functions = [damped_sine]
    elif dim == 2:
        functions = [damped_sine, cubic_func]
    elif dim == 3:
        functions = [damped_sine, cubic_func, freq_increasing_sine]
    else:
        assert False

    series = generate_time_series(dim=dim, t_range=t_range, count=count,
                                      functions=functions, anomaly_rate=anomaly_rate,
                                      add_noise=True, noise_var=0.01)

    return series
