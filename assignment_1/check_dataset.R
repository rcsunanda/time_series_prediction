# Load data and plot
training_data <- read.csv('data/train.csv')
store_aggregated_train = aggregate(sales ~ date + item, training_data, sum)

item_1 = store_aggregated_train[store_aggregated_train$item == 1, ]
item_1_ts <- ts(item_1[,3], start=c(2013,1,1), frequency=365)
plot(item_1_ts, xlab='Time', ylab='Item 1 Sales', xlim=c(2013,2014))

# Plot diff to check whether series is stationary (no variance around trend) --> this is not stationary
plot(diff(item_1_ts), ylab='Differenced', xlim=c(2013,2014))

# log is not necessary
# # Plot diff of log to see if log is stationary --> yes it's stationary
# plot(diff(log10(item_1_ts)), ylab='Differenced', xlim=c(2013,2014))

# Plot ACF and PACF to identify AR and MA patterns in the residuals
# They are present (as well as seasonality with period 12)
par(mfrow=c(1,2))
acf(ts(diff(item_1_ts)), main='ACF')
pacf(ts(diff(item_1_ts)), main='PACF')

# Train an ARIMA model
require(forecast)
print("Starting ARIMA")
Sys.time()
ARIMAfit <- auto.arima(item_1_ts)
print("Finished ARIMA")
Sys.time()
summary(ARIMAfit)


test_data <- read.csv('data/test.csv')
store_aggregated_test = aggregate(sales ~ date + item, test_data, sum)
item_1_test = store_aggregated_test[store_aggregated_test$item == 1, ]
item_1_test_ts <- ts(item_1_test[,3], start=c(2018,1,1), frequency=365)

# Predict some future values in time series using the trained ARIMA model
par(mfrow = c(1,1))
pred <- predict(ARIMAfit, n.ahead=300)

# Plot actual time series, prediction and their 2*sigma confidence interval bounds
plot(item_1_ts, type='l', xlim=c(2015, 2019), xlab='Year', ylab='Item 1 Sales')
lines(pred$pred, col='blue')
lines(pred$pred+2*pred$se, col='orange')
lines(pred$pred-2*pred$se, col='orange')

# lines(10^(pred$pred),col='blue')
# lines(10^(pred$pred+2*pred$se),col='orange')
# lines(10^(pred$pred-2*pred$se),col='orange')
# 
# Check the ACF and PACF plots of the residuals of our best fit ARIMA model
# Verify that only random noise is remaining (no significant parts)
par(mfrow=c(1,2))
acf(ts(ARIMAfit$residuals), main='ACF Residual')
pacf(ts(ARIMAfit$residuals), main='PACF Residual')
