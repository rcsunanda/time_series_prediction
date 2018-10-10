######### Loading and Visualization ##########

# Load data and plot
all_data <- read.csv('data/train.csv')
store_aggregated_data = aggregate(sales ~ date + item, training_data, sum)

item_1_data = store_aggregated_data[store_aggregated_data$item == 1, c(1,3)]
data_split_index = ceiling(nrow(item_1_data)*0.8)
item_1_train <- item_1_data[1:data_split_index,]
item_1_test <- item_1_data[-(1:data_split_index),]

item_1_train_ts <- ts(item_1_train[,2], start=c(2013,1,1), frequency=365)
item_1_test_ts <- ts(item_1_test[,2], start=c(2017,1,1), frequency=365)
plot(item_1_train_ts, xlab='Time', ylab='Item 1 Sales', xlim=c(2013,2016))


######### ARIMA Checking ##########

# Plot diff to check whether series is stationary (no variance around trend) --> this is not stationary
plot(diff(item_1_train_ts), ylab='Differenced', xlim=c(2013,2014))

# log is not necessary
# # Plot diff of log to see if log is stationary --> yes it's stationary
# plot(diff(log10(item_1_train_ts)), ylab='Differenced', xlim=c(2013,2014))

# Plot ACF and PACF to identify AR and MA patterns in the residuals
# They are present (as well as seasonality with period 12)
par(mfrow=c(1,2))
acf(ts(diff(item_1_train_ts)), main='ACF')
pacf(ts(diff(item_1_train_ts)), main='PACF')


######### ARIMA Training ##########

# Train an ARIMA model
require(forecast)
print("Starting training ARIMA")
Sys.time()
ARIMAfit <- auto.arima(item_1_train_ts)
print("Finished training ARIMA")
Sys.time()
summary(ARIMAfit)


######### ARIMA Testing ##########

# Predict some future values in time series using the trained ARIMA model
par(mfrow = c(1,1))
pred <- predict(ARIMAfit, n.ahead=365)

# Plot actual time series, prediction and their 2*sigma confidence interval bounds
plot(item_1_train_ts, type='l', xlim=c(2013, 2019), xlab='Year', ylab='Item 1 Sales') # Plot training data
lines(item_1_test_ts, col='red') # Plot actual values of test data
lines(pred$pred, col='blue')  # Plot predicted values of test data
lines(pred$pred+2*pred$se, col='orange')  # Plot prediction intervals
lines(pred$pred-2*pred$se, col='orange')


######### ARIMA Post Analysis ##########

# Check the ACF and PACF plots of the residuals of our best fit ARIMA model
# Verify that only random noise is remaining (no significant parts)
par(mfrow=c(1,2))
acf(ts(ARIMAfit$residuals), main='ACF Residual')
pacf(ts(ARIMAfit$residuals), main='PACF Residual')
