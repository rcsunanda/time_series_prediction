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


######### Build KNN and predict ##########

library(tsfknn)
library(ggplot2)

knn_pred <- knn_forecasting(item_1_train_ts, h=365, lags=c(1:10, 355:365), k=30)
# knn_examples(pred)

plot(item_1_train_ts, type='l', xlim=c(2016, 2018), xlab='Year', ylab='Item 1 Sales') # Plot training data
lines(item_1_test_ts, col='red') # Plot actual values of test data
lines(knn_pred$prediction, col='blue') # Plot predicted values of test data

legend(2015.95, 390, legend=c("Training data", "Actual test data", "KNN prediction"),
       col=c("black", "red", "blue"), lty=1:2, cex=0.8)
plot(knn_pred)
autoplot(knn_pred, highlight="neighbors")

library(DMwR)
ts.eval(item_1_test_ts, knn_pred$prediction)
