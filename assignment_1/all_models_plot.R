# This script must be run after running and predicting with the 3 models

# Actual test vals
plot(item_1_train_ts, type='l', xlim=c(2017, 2018), xlab='Year', ylab='Item 1 Sales') # Plot training data
lines(item_1_test_ts, col='red') # Plot actual values of test data

# ARIMA
lines(pred$pred, col='blue')

# KNN
lines(knn_pred$prediction, col='orange')

# LTSM
lines(ltsm_predictions_ts, col='green')


legend(2016.95, 385, legend=c("Actual test data", "ARIMA prediction", "Confidence intervals"),
       col=c("red", "blue", "orange", "green"), lty=1:2, cex=0.8)