
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


def generate_high_dim_complex_series(dim, t_range, count, anomaly_rate=0):

    def damped_sine(x, d):
        return (d+1) * (math.e**x) * (math.sin(2 * math.pi * x))

    def cubic_func(x, d):
        return (d+1) * x**3 - (d+1) * 6 * x**2 + 4*x + 12

    def freq_increasing_sine(x, d):
        return (d+1) * math.sin(2 * math.pi * math.e**x)

    all_functions = [damped_sine, cubic_func, freq_increasing_sine]

    functions = []
    for d in range(dim):
        func_index = d % 3
        functions.append(all_functions[func_index])

    series = generate_time_series(dim=dim, t_range=t_range, count=count,
                                      functions=functions, anomaly_rate=anomaly_rate,
                                      add_noise=True, noise_var=0)  #noise_var=0.005

    return series


# Generate time series of given size
def generate_time_series(dim, t_range, count, functions, anomaly_rate=0, add_noise=False, noise_var=1):
    assert (len(t_range) == 2)
    assert (len(functions) == dim)

    t_vals = np.linspace(t_range[0], t_range[1], num=count)
    series = []
    # Sample value for each dimension
    for idx in range(count):
        sample_X = np.zeros((dim,))
        t = t_vals[idx]

        for d in range(dim):
            func = functions[d]
            dim_sample_val = func(t, d)
            sample_X[d] = dim_sample_val

        if add_noise:
            noise_vec = np.random.multivariate_normal(np.zeros(dim), np.eye(dim)*noise_var)
            sample_X = np.add(sample_X, noise_vec)

        series.append(data_point.DataPoint(t, sample_X, False, False))

    # Set anomolous points

    anomaly_count = int(count * anomaly_rate)
    anomalous_point_indexes = np.random.randint(low=0, high=count, size=anomaly_count)
    for an_idx in anomalous_point_indexes:
        point = series[an_idx]

        low = np.add(point.X, np.ones(dim))
        high = np.subtract(point.X, np.ones(dim))
        deviation_vec = np.random.uniform(low, high)

        point.X = np.add(point.X, deviation_vec)
        point.true_is_anomaly = True

    return series
