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
# plot(item_1_train_ts, xlab='Time', ylab='Item 1 Sales', xlim=c(2013,2016))

# Read LTSM predictions

ltsm_predictions <- read.csv('data/ltsm_predicted.csv')
ltsm_predictions_ts <- ts(ltsm_predictions[,2], start=c(2017,1,1), frequency=365)

library(DMwR)
ts.eval(item_1_test_ts, ltsm_predictions_ts)

plot(item_1_test_ts) # Plot testing data
lines(ltsm_predictions_ts, col='red')
