#Main
if(!require("ggplot2")) {
  install.packages("ggplot2")
  library("ggplot2")
}
if(!require("keras")) {
  install.packages("keras")
  library("keras")
}
if(!require("tensorflow")) {
  install.packages("tensorflow")
  library("tensorflow")
}
install_keras()
install_tensorflow(version="nightly")

#Create Data Sets
source("DataCreation.R")
anomaly2<-c(normal(10,1,30),DataCreation(10,1,1,10,"cyclic"))

#Split Data
train<-anomaly2[1:20]
test<-anomaly2[21:30]
test2<-anomaly2[31:40]

#Scale Data
scale_factors_train <- c(mean(train), sd(train))
scaled_train <-scale(train)
scale_factors_test <- c(mean(test), sd(test))
scaled_test<-scale(test)
scale_factors_test2 <- c(mean(test2), sd(test2))
scaled_test2<-scale(test2)

#Structure data for lstm
n_steps<-3

# transform into [batch_size, timesteps, features] format required by RNNs
gen_timesteps <- function(x, n_timesteps) {
  do.call(rbind,
          purrr::map(seq_along(x),
                     function(i) {
                       start <- i
                       end <- i + n_timesteps - 1
                       out <- x[start:end]
                       out
                     })
  ) %>%
    na.omit()
}


train <- gen_timesteps(scaled_train, 2 * n_steps)
test <- gen_timesteps(scaled_test, 2 * n_steps) 

dim(train) <- c(dim(train), 1)
dim(test) <- c(dim(test), 1)

# split into input and target  
x_train <- train[ , 1:n_steps, , drop = FALSE]
y_train <- train[ , (n_steps + 1):(2*n_steps), , drop = FALSE]

x_test <- test[ , 1:n_steps, , drop = FALSE]
y_test <- test[ , (n_steps + 1):(2*n_steps), , drop = FALSE]


#train model
source("LSTMmodel.R")
lstm_model %>% fit(
  x = x_train,
  y = y_train,
  batch_size = 1,
  epochs = 20,
  verbose = 0,
  shuffle = FALSE
)

#predict
lstm_forecast <- lstm_model %>%
  predict(x_train, batch_size = 1) %>%
  .[, , 1]

# we need to rescale the data to restore the original values
lstm_forecast <- lstm_forecast * scale_factors[2] + scale_factors[1]