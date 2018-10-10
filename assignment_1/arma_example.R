# Load data and plot
raw_data <- read.csv('http://ucanalytics.com/blogs/wp-content/uploads/2015/06/Tractor-Sales.csv')
data <- ts(raw_data[,2], start=c(2003,1),frequency=12)
plot(data, xlab='Years', ylab='Tractor Sales')

# Plot diff to check whether series is stationary (no variance around trend) --> this is not stationary
plot(diff(data), ylab='Differenced Tractor Sales')

# Plot diff of log to see if log is stationary --> yes it's stationary
plot(diff(log10(data)),ylab='Differenced Log (Tractor Sales)')

# Plot ACF and PACF to identify AR and MA patterns in the residuals
# They are present (as well as seasonality with period 12)
par(mfrow=c(1,2))
acf(ts(diff(log10(data))), main='ACF Tractor Sales')
pacf(ts(diff(log10(data))), main='PACF Tractor Sales')

# Train an ARIMA model
require(forecast)
ARIMAfit <- auto.arima(log10(data), approximation=FALSE, trace=FALSE)
summary(ARIMAfit)

# Predict some future values in time series using the trained ARIMA model
par(mfrow = c(1,1))
pred <- predict(ARIMAfit, n.ahead=36)

# Plot actual time series, prediction and their 2*sigma confidence interval bounds
plot(data, type='l', xlim=c(2004,2018), ylim=c(1,1600), xlab='Year', ylab='Tractor Sales')
lines(10^(pred$pred),col='blue')
lines(10^(pred$pred+2*pred$se),col='orange')
lines(10^(pred$pred-2*pred$se),col='orange')

# Check the ACF and PACF plots of the residuals of our best fit ARIMA model
# Verify that only random noise is remaining (no significant parts)
par(mfrow=c(1,2))
acf(ts(ARIMAfit$residuals),main='ACF Residual')
pacf(ts(ARIMAfit$residuals),main='PACF Residual')
