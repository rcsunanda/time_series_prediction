"""
DataPoint class
"""


###################################################################################################
"""
Holds (X, true_is_anomaly, predicted_is_anomaly) for each sample or data point (of a time series)
"""

class DataPoint:
    def __init__(self, t, X, true_is_anomaly, predicted_is_anomaly):
        self.t = t  # Time index
        self.X = X
        self.true_is_anomaly = true_is_anomaly
        self.predicted_is_anomaly = predicted_is_anomaly

    def __repr__(self):
        return "DataPoint(\n\tX={} \n\ttrue_is_anomaly={} \n\tpredicted_is_anomaly={}\n)".\
            format(self.X, self.true_is_anomaly, self.predicted_is_anomaly)
