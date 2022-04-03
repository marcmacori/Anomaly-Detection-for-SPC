lstm_model <- keras_model_sequential()

lstm_model %>%
  layer_lstm(units = 50, # size of the layer
             batch_input_shape = c(1, 3, 1), # batch size, timesteps, features
             return_sequences = TRUE,
             stateful = TRUE) %>%
  # fraction of the units to drop for the linear transformation of the inputs
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 50,
             return_sequences = TRUE,
             stateful = TRUE) %>%
  layer_dropout(rate = 0.5) %>%
  time_distributed(keras::layer_dense(units = 1)) %>%
  compile(loss = "mae", optimizer = "adam", metrics = "accuracy")

summary(lstm_model)