# Build and evaluate neural networks for the estimation of physiological tissue
# forces from inertial measurement unit data

# Written by Joseph Shaw (shaw_joseph@outlook.com) 

# Tuning run?
tuning_run <- 0

# select target (Achilles tendon = 1, patellar tendon = 2, tibia = 3, GRF = 4)
target <- 1

# select imus (All = 1, Waist/Shank = 2)
imus <- 1

# Load packages -----------------------------------------------------------

library(tidyverse)
library(tibble)
library(readr)
library(keras)
#remotes::install_github("elliefewings/multigrep")
library(multigrep)
library(abind)
library(tfruns)
library(patchwork)


# Set seed ----------------------------------------------------------------

set.seed(1234)
tensorflow::set_random_seed(1234)

# Load data ---------------------------------------------------------------

# create list of 5 participants and validation participant
participants <- c("p01", "p02", "p03", "p04", "p05", "p06")
folder <- "data/"

# loop through data for each participant
for(i in participants){
  
  # create file path
  file <- paste0(folder, i, "_data.csv")
  
  # read file
  data <- read.csv(file) 
  
  # add col to identify tuning validation set
  data$tune <- 0
  data$tune[sample(1:nrow(data), size = nrow(data)*0.05, replace = F)] <- 1
  
  
  # assign to participant reference in global env
  assign(i, data)
  
}

# bind each participant's data
data_orig <- rbind(p01, p02, p03, p04, p05, p06)

for(t in 1:6){
  
  # give new participant id, assigning each participant as the test in turn
  train_p <- which(1:6 != t)
  
  # create index of each participant
  p01.id <- which(data_orig$participant == train_p[1])
  p02.id <- which(data_orig$participant == train_p[2])
  p03.id <- which(data_orig$participant == train_p[3])
  p04.id <- which(data_orig$participant == train_p[4])
  p05.id <- which(data_orig$participant == train_p[5])
  pTe.id <- which(data_orig$participant == t)
  
  # make vector of anthropometry column indexes
  col.ids <- which(multigrep::multigrep(c("com", "mass", "long", "sag", "tran"), names(data_orig)))
  
  # drop columns
  data <- data_orig %>% 
    select(-col.ids) %>%   # remove anthro columns
    select(-c(1:7))        # remove index columns
  
  # make a vector of targets for easy removal later
  targets <- which(names(data) %in% c("achilles", "anklex", "ankley", "anklez", "patellartendon", "fx", "fy", "fz", "grf", "tibia"))
  
  # divide into train / test / validate
  data.train <- data[-pTe.id,]
  data.test <- data[pTe.id,]
  ws.data_orig <- data_orig[-pTe.id,]
  
  # Scale data --------------------------------------------------------------
  
  # create function to normalize data
  normalize <- function(x, y) {
    return ((x - min(y, na.rm = T)) / (max(y, na.rm = T) - min(y, na.rm = T)))
  }
  
  # scale
  mean <- apply(data.train, 2, mean, na.rm = T)
  std <- apply(data.train, 2, sd, na.rm = T)
  data.train <- as.data.frame(scale(data.train, center = mean, scale = std)) #scaled data frames
  data.test <- as.data.frame(scale(data.test, center = mean, scale = std))

  # copy of data train for test and val normalization
  data.ref <- data.train
  
  # normalize each column based on training data
  for(i in 1:ncol(data.train)){
    data.train[,i] <- normalize(data.train[,i], y = data.ref[,i])
    data.test[,i] <- normalize(data.test[,i], y = data.ref[,i])
  }
  
  # re-add limb data
  data.train$limb <- data_orig$limb[-c(pTe.id)]-1 
  data.test$limb  <- data_orig$limb[pTe.id]-1 
  
  # Create time windows -----------------------------------------------------
  
  # re-separate data before creating time windows
  p01.data <- data.train[p01.id,]
  p02.data <- data.train[p02.id,]
  p03.data <- data.train[p03.id,]
  p04.data <- data.train[p04.id,]
  p05.data <- data.train[p05.id,]
  pTe.data <- data.test
  ws.data <- data.train
  ws.data$set_move <- data_orig$set_move[-c(pTe.id)]
  
  # parameters of time window
  
  if(
    target == 1 | 
    (target == 4 & imus == 1)
    ){lookback <- 4}else{lookback <- 5}
  if(
    (target == 1 & imus == 1) |
    (target == 2) |
    (target == 4 & imus == 2) |
    (target == 3 & imus == 1)
  ){delay <- 2}else{delay <- 1}
  
  step <- 1
  min_index <- 1
  
  # create function to organise data into nn inputs (rolling time windows) and outputs 
  create.ts <- function(dat = dat, lookback, delay, min_index, max_index,
                        step = 1, ach = T, pat = T, tib = T, grf = T) {
    function() {
      i <<- min_index + lookback
      rows <<- c(i:min(i+max_index-1, max_index))
      i <<- i + length(rows)
      
      samples <- array(0, dim = c(length(rows),
                                  lookback / step,
                                  dim(dat)[[-1]]))
      targets_pat <- array(0, dim = c(length(rows)))
      targets_ach <- array(0, dim = c(length(rows)))
      targets_tib <- array(0, dim = c(length(rows)))
      targets_grf <- array(0, dim = c(length(rows)))
      
      
      for (j in 1:length(rows)) {
        indices <- seq(rows[[j]] + delay, rows[[j]] - lookback + 1 + delay,
                       length.out = dim(samples)[[2]])
        indices = rev(indices) ###
        samples[j,,] <- data.matrix(dat[indices,])
        
        if(tib){
          targets_tib[[j]] <- data.matrix(dat[rows[[j]]+1, 65])   
        }
        if(pat){
          targets_pat[[j]] <- data.matrix(dat[rows[[j]]+1, 60])    
        }
        if(ach){
          targets_ach[[j]] <- data.matrix(dat[rows[[j]]+1, 56])   
        }
        if(grf){
          targets_grf[[j]] <- data.matrix(dat[rows[[j]]+1, 64])   
        }
      }         
      list(samples, 
           if(ach){targets_ach},
           if(tib){targets_tib},
           if(pat){targets_pat},
           if(grf){targets_grf}
      )
    }
  }
  

# Create rolling windows for each participant -----------------------------

  #p01
  dat <- p01.data
  max_index <- nrow(p01.data)
  data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                           min_index = 1, max_index = max_index)
  data_rnn_p01 <- data_rnn_ts()
  
  #p02
  dat <- p02.data
  max_index <- nrow(p02.data)
  data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                           min_index = 1, max_index = max_index)
  data_rnn_p02 <- data_rnn_ts()
  #p03
  dat <- p03.data
  max_index <- nrow(p03.data)
  data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                           min_index = 1, max_index = max_index)
  data_rnn_p03 <- data_rnn_ts()
  #p04
  dat <- p04.data
  max_index <- nrow(p04.data)
  data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                           min_index = 1, max_index = max_index)
  data_rnn_p04 <- data_rnn_ts()
  #p05
  dat <- p05.data
  max_index <- nrow(p05.data)
  data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                           min_index = 1, max_index = max_index)
  data_rnn_p05 <- data_rnn_ts()
  # test
  dat <- data.test
  max_index <- nrow(data.test)
  data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                           min_index = 1, max_index = max_index)
  data_rnn_pTe <- data_rnn_ts()
  
  # Bind time windows -------------------------------------------------------
  
  # ~training
  # bind windows
  data_train_x <-   abind(data_rnn_p01[[1]], data_rnn_p02[[1]], data_rnn_p03[[1]], data_rnn_p04[[1]], data_rnn_p05[[1]], along = 1)
  data_train_ach <- abind(data_rnn_p01[[2]], data_rnn_p02[[2]], data_rnn_p03[[2]], data_rnn_p04[[2]], data_rnn_p05[[2]], along = 1)
  data_train_tib <- abind(data_rnn_p01[[3]], data_rnn_p02[[3]], data_rnn_p03[[3]], data_rnn_p04[[3]], data_rnn_p05[[3]], along = 1)
  data_train_pat <- abind(data_rnn_p01[[4]], data_rnn_p02[[4]], data_rnn_p03[[4]], data_rnn_p04[[4]], data_rnn_p05[[4]], along = 1)
  data_train_grf <- abind(data_rnn_p01[[5]], data_rnn_p02[[5]], data_rnn_p03[[5]], data_rnn_p04[[5]], data_rnn_p05[[5]], along = 1)
  
  # remove rows with NAs (and remove targets)
  drop.rows <- which(is.na(data_train_ach))
  data_train_x <-   data_train_x[-drop.rows,,-targets]
  data_train_ach <- data_train_ach[-drop.rows]
  data_train_tib <- data_train_tib[-drop.rows]
  data_train_pat <- data_train_pat[-drop.rows]
  data_train_grf <- data_train_grf[-drop.rows]
  
  
  nas <- apply(data_train_x, 1, is.na)
  na.rows <- apply(nas, 2, max) == 0
  data_train_x <-   data_train_x[na.rows,,]
  data_train_ach <- data_train_ach[na.rows]
  data_train_tib <- data_train_tib[na.rows]
  data_train_pat <- data_train_pat[na.rows]
  data_train_grf <- data_train_grf[na.rows]
  
  # if a tuning run, remove rows from training set to use as validation. If not, leave in for training.
  
  tune.id <- 56 #ifelse(imus == 1, 56, 34)
  train.tune.id <- which(data_train_x[,lookback+1-delay,tune.id] == 1)
  
  if(tuning_run == 1){
    
  data_tune_x <-   data_train_x[train.tune.id,,]
  data_tune_ach <- data_train_ach[train.tune.id]
  data_tune_tib <- data_train_tib[train.tune.id]
  data_tune_pat <- data_train_pat[train.tune.id]
  data_tune_grf <- data_train_grf[train.tune.id]  
    
  data_train_x <-   data_train_x[-train.tune.id,,]
  data_train_ach <- data_train_ach[-train.tune.id]
  data_train_tib <- data_train_tib[-train.tune.id]
  data_train_pat <- data_train_pat[-train.tune.id]
  data_train_grf <- data_train_grf[-train.tune.id]  
  
  }
  
  data_test_x <-   data_rnn_pTe[[1]][,,-targets]
  
  # identify tuning rows
  test.tune.id <- which(data_test_x[,lookback+1-delay,tune.id] == 1)
  data_test_x <- data_test_x[-test.tune.id,,]
  
  # select target
  if(target == 1){data_test_tar <- data_rnn_pTe[[2]]}
  if(target == 2){data_test_tar <- data_rnn_pTe[[4]]}
  if(target == 3){data_test_tar <- data_rnn_pTe[[3]]}
  if(target == 4){data_test_tar <- data_rnn_pTe[[5]]}
  
  if(target == 1){data_train_tar <- data_train_ach}
  if(target == 2){data_train_tar <- data_train_pat}
  if(target == 3){data_train_tar <- data_train_tib}
  if(target == 4){data_train_tar <- data_train_grf}
  
  # remove tuning rows
  data_test_tar <- data_test_tar[-test.tune.id]
  
  # if not a tuning run, divide training into training and validation
  if(tuning_run == 0){
  rows <- sample(1:nrow(data_train_ach), ceiling(0.75*nrow(data_train_ach)), replace = F)
  data_val_x <- data_train_x[-rows,,]
  data_train_x <- data_train_x[rows,,]
  if(target == 1){
    data_val_tar <- data_train_ach[-rows]
    data_train_tar <- data_train_ach[rows]
  }  
  if(target == 2){
    data_val_tar <- data_train_pat[-rows]
    data_train_tar <- data_train_pat[rows]
  }  
  if(target == 3){
    data_val_tar <- data_train_tib[-rows]
    data_train_tar <- data_train_tib[rows]
  }  
  if(target == 4){
    data_val_tar <- data_train_grf[-rows]
    data_train_tar <- data_train_grf[rows]
  }  
  
  }else{
    data_val_x <- data_tune_x
    if(target == 1){data_val_tar <- data_tune_ach}
    if(target == 2){data_val_tar <- data_tune_pat}
    if(target == 3){data_val_tar <- data_tune_tib}
    if(target == 4){data_val_tar <- data_tune_grf}
  }
  
  
  

# Resample data -----------------------------------------------------------
# conduct bootstrap resampling to balance the distribution of target values
# exclude 0.00 - 0.05 to stop imbalance prediction
  limb <- which(names(dat) == "limb")
  bands <- c(seq(0.05, 1, 0.05))    
  for(i in 1:19){
    if(!is.null(which(data_train_tar >= bands[i] & data_train_tar <= bands[i+1] & data_train_x[,lookback,dim(data_train_x)[3]] == 1))){
      id <-    which(data_train_tar >= bands[i] & data_train_tar <= bands[i+1] & data_train_x[,lookback,dim(data_train_x)[3]] == 1)}
    if(!is.null(which(data_val_tar >= bands[i] &   data_val_tar <= bands[i+1] &   data_val_x[,lookback,dim(data_train_x)[3]] == 1))){
      id.v <-  which(data_val_tar >= bands[i] &   data_val_tar <= bands[i+1] &   data_val_x[,lookback,dim(data_train_x)[3]] == 1)}
    if(!is.null(which(data_train_tar >= bands[i] & data_train_tar <= bands[i+1] & data_train_x[,lookback,dim(data_train_x)[3]] == 0))){
      id1 <-   which(data_train_tar >= bands[i] & data_train_tar <= bands[i+1] & data_train_x[,lookback,dim(data_train_x)[3]] == 0)}
    if(!is.null(which(data_val_tar >= bands[i] &   data_val_tar <= bands[i+1] &   data_val_x[,lookback,dim(data_train_x)[3]] == 0))){
      id.v1 <- which(data_val_tar >= bands[i] &   data_val_tar <= bands[i+1] &   data_val_x[,lookback,dim(data_train_x)[3]] == 0)}
    
    if(i == 1){
      resample.train2 <- sample(id, 3000, replace = T)
      resample.train1 <- sample(id1, 3000, replace = T)
      resample.val2 <- sample(id.v, 1000, replace = T)
      resample.val1 <- sample(id.v1, 1000, replace = T)
    }else{
      if(!is_empty(id)){
        resample.train2 <- c(resample.train2, sample(id, 3000, replace = T))
      }
      if(!is_empty(id1)){
        resample.train1 <- c(resample.train1, sample(id1, 3000, replace = T))
      }
      if(!is_empty(id.v)){
        resample.val2 <- c(resample.val2, sample(id.v, 1000, replace = T))
      }
      if(!is_empty(id.v1)){
        resample.val1 <- c(resample.val1, sample(id.v1, 1000, replace = T))
      }
    }
  }
  
  res_tr <- c(resample.train1, resample.train2)
  res_va <- c(resample.val1, resample.val2)
  
  data_train_x_r <- data_train_x[res_tr,,]
  data_train_tar_r <- data_train_tar[res_tr]
  data_val_x_r <-   data_val_x[res_va,,]
  data_val_tar_r <- data_val_tar[res_va]

  
  # Remove unwanted features ------------------------------------------------
  
  foot <- which(grepl("foot", names(dat[,-targets]), ignore.case = T))
  shank <- which(grepl("shank", names(dat[,-targets]), ignore.case = T))
  thigh <- which(grepl("thigh", names(dat[,-targets]), ignore.case = T))
  waist <- which(grepl("waist", names(dat[,-targets]), ignore.case = T))
  scap <- which(grepl("scap", names(dat[,-targets]), ignore.case = T))
  
  
  if(imus == 2){
  drop <- c(foot, thigh, scap)
  }else{if(imus == 1){
    drop <- c(NULL)
  }}
  
  if(!is.null(drop)){
    data_val_x <- data_val_x[,,-c(drop)]
    data_val_x_r <- data_val_x_r[,,-c(drop)]
    data_train_x_r <- data_train_x_r[,,-c(drop)]
    data_test_x <- data_test_x[,,-c(drop)]
  }
  
  drop <- ifelse(imus == 1, 56, 23)
  # drop tuning id slice
  data_train_x_r <- data_train_x_r[,,-23]
  data_val_x_r <- data_val_x_r[,,-23]
  data_test_x <- data_test_x[,,-23]
  
  # shuffle data
  rows <- sample(nrow(data_train_x_r), replace = F)
  
  data_train_x_r <- data_train_x_r[rows, ,]
  data_train_tar_r <- data_train_tar_r[rows]
  

  
  
  
# Model 1: Achilles Force | All IMUs --------------------------------------
  # 4data points, 2 delay
if(target == 1 & imus == 1){
  
  # build model
  model <- keras_model_sequential() %>%
    layer_lstm(units = 64,
               input_shape = c(dim(data_train_x_r)[-1]), 
               batch_size = NULL,              
               return_sequences = F,
               recurrent_dropout = 0.4
    ) %>%
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear") 
  
  early_stopping <- keras::callback_early_stopping(
    monitor = "val_loss",
    patience = 15,
    min_delta = 0.0001,
    restore_best_weights = T)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.000005),
    loss = "mse"
  )
  
  # train model
  history <- model %>% fit(
    x = data_train_x_r,
    y = data_train_tar_r,
    validation_data = list(
      data_val_x_r,
      data_val_tar_r
    ),
    epochs = 400,
    shuffle = T,
    callbacks = early_stopping
  ) 
  
}
  

# Model 2: Achilles Force | Waist & Shank ---------------------------------
  # 4 data points, 1 delay
  
  if(target == 1 & imus == 2){
    
    # build model
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64,
                 input_shape = c(dim(data_train_x_r)[-1]), 
                 batch_size = NULL,              
                 return_sequences = F,
                 recurrent_dropout = 0.4
      ) %>% 
      layer_dense(units = 64, activation = "relu")%>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear") 
    
    early_stopping <- keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      min_delta = 0.0001,
      restore_best_weights = T)
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.00001),
      loss = "mse"
    )
    
    # train model
    history <- model %>% fit(
      x = data_train_x_r,
      y = data_train_tar_r,
      validation_data = list(
        data_val_x_r,
        data_val_tar_r
      ),
      epochs = 400,
      shuffle = T,
      callbacks = early_stopping
    ) 
  }

# Model 3: PT Force | All IMUs --------------------------------------------
  # 5 and 2
  
if(target == 2 & imus == 1){
  
  # build model
  model <- keras_model_sequential() %>%
    layer_lstm(units = 64,
               input_shape = c(dim(data_train_x_r)[-1]), 
               batch_size = NULL,              
               return_sequences = F,
               recurrent_dropout = 0.4
    ) %>% 
    layer_dense(units = 64, activation = "relu")%>% 
    layer_dense(units = 32, activation = "relu") %>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear") 
  
    early_stopping <- keras::callback_early_stopping(
    monitor = "val_loss",
    patience = 15,
    min_delta = 0.0001,
    restore_best_weights = T)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.00001),
    loss = "mse"
  )
  
  # train model
  history <- model %>% fit(
    x = data_train_x_r,
    y = data_train_tar_r,
    validation_data = list(
      data_val_x_r,
      data_val_tar_r
    ),
    epochs = 400,
    shuffle = T,
    callbacks = early_stopping
  ) 
  
}
  

# Model 4: PT Force | Waist & Shank ---------------------------------------
  # 5 and 2
  
  if(target == 2 & imus == 2){
    # build model
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64,
                 input_shape = c(dim(data_train_x_r)[-1]), 
                 batch_size = NULL,              
                 return_sequences = F,
                 recurrent_dropout = 0.4
      )%>% 
      layer_dense(units = 64, activation = "relu")%>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear") 
    
    early_stopping <- keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      min_delta = 0.0001,
      restore_best_weights = T)
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.00001),
      loss = "mse"
    )
    
    # train model
    history <- model %>% fit(
      x = data_train_x_r,
      y = data_train_tar_r,
      validation_data = list(
        data_val_x_r,
        data_val_tar_r
      ),
      epochs = 400,
      shuffle = T,
      callbacks = early_stopping
    ) 
    
  }

# Model 5: Tibial Force | All IMUs ---------------------------------------
  # 5 and 2
  
  if(target == 3 & imus == 1){
    # build model
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64,
                 input_shape = c(dim(data_train_x_r)[-1]), 
                 batch_size = NULL,              
                 return_sequences = F,
                 recurrent_dropout = 0.4
      ) %>%
      layer_dense(units = 64, activation = "relu")%>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear") 
    
    early_stopping <- keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      min_delta = 0.0001,
      restore_best_weights = T)
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.00001),
      loss = "mse"
    )
    
    # train model
    history <- model %>% fit(
      x = data_train_x_r,
      y = data_train_tar_r,
      validation_data = list(
        data_val_x_r,
        data_val_tar_r
      ),
      epochs = 400,
      shuffle = T,
      callbacks = early_stopping
    ) 
    
  }  
  
# Model 6: Tibial Force | Waist & Shank ---------------------------------------
  # 5 and 1 
  
  if(target == 3 & imus == 2){
    # build model
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64,
                 input_shape = c(dim(data_train_x_r)[-1]), 
                 batch_size = NULL,              
                 return_sequences = F,
                 recurrent_dropout = 0.4
      ) %>%
      layer_dense(units = 64, activation = "relu") %>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear") 
    
    early_stopping <- keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      min_delta = 0.0001,
      restore_best_weights = T)
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.00001),
      loss = "mse"
    )
    
    # train model
    history <- model %>% fit(
      x = data_train_x_r,
      y = data_train_tar_r,
      validation_data = list(
        data_val_x_r,
        data_val_tar_r
      ),
      epochs = 400,
      shuffle = T,
      callbacks = early_stopping
    ) 
    
  }  
  
# Model 7: GRF | All IMUs -------------------------------------------------
# 4 and 1
  if(target == 4 & imus == 1){
    # build model
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64,
                 input_shape = c(dim(data_train_x_r)[-1]), 
                 batch_size = NULL,              
                 return_sequences = F,
                 recurrent_dropout = 0.4
      ) %>% 
      layer_dense(units = 64, activation = "relu")%>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear") 
    
    early_stopping <- keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      min_delta = 0.0001,
      restore_best_weights = T)
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.000001),
      loss = "mse"
    )
    
    # train model
    history <- model %>% fit(
      x = data_train_x_r,
      y = data_train_tar_r,
      validation_data = list(
        data_val_x_r,
        data_val_tar_r
      ),
      epochs = 400,
      shuffle = T,
      callbacks = early_stopping
    ) 
    
  }
  

# Model 8: GRF | Waist & Shank --------------------------------------------
#5 and 2
  if(target == 4 & imus == 2){
    # build model
    model <- keras_model_sequential() %>%
      layer_lstm(units = 64,
                 input_shape = c(dim(data_train_x_r)[-1]), 
                 batch_size = NULL,              
                 return_sequences = F,
                 recurrent_dropout = 0.4
      )%>% 
      layer_dense(units = 64, activation = "relu")%>% 
      layer_dense(units = 32, activation = "relu") %>% 
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear") 
    
    early_stopping <- keras::callback_early_stopping(
      monitor = "val_loss",
      patience = 15,
      min_delta = 0.0001,
      restore_best_weights = T)
    
    model %>% compile(
      optimizer = optimizer_adam(learning_rate = 0.00001),
      loss = "mse"
    )
    
    # train model
    history <- model %>% fit(
      x = data_train_x_r,
      y = data_train_tar_r,
      validation_data = list(
        data_val_x_r,
        data_val_tar_r
      ),
      epochs = 400,
      shuffle = T,
      callbacks = early_stopping
    ) 
  }

  
  model %>% save_model_tf("model")

# Waist and shank - Linear transformation ---------------------------------

  if(target == 1){lm.tar <- ws.data$achilles}
  if(target == 2){lm.tar <- ws.data$patellartendon}
  if(target == 3){lm.tar <- ws.data$tibia}
  if(target == 4){lm.tar <- ws.data$grf}
  
  waist.mod <- lm(lm.tar ~ WAIST_r.acc, data = ws.data)  
  shank.mod <- lm(lm.tar ~ SHANK_r.acc, data = ws.data)  

  if(target == 1){lm.tar <- ws.data_orig$achilles}
  if(target == 2){lm.tar <- ws.data_orig$patellartendon}
  if(target == 3){lm.tar <- ws.data_orig$tibia}
  if(target == 4){lm.tar <- ws.data_orig$grf}
  
  ws_pk.data <- ws.data_orig %>% 
    mutate(
      target = lm.tar
    ) %>% 
    group_by(set_move) %>% 
    summarise(
      waist = max(WAIST_r.acc, na.rm = T),
      shank = max(SHANK_r.acc, na.rm = T),
      target = max(target, na.rm = T)
    ) %>% 
    arrange(target) %>% 
    filter(
      !grepl("na", set_move, ignore.case = T)
    )
  
  waist_pk.mod <- lm(target ~ waist, data = ws_pk.data)  
  shank_pk.mod <- lm(target ~ shank, data = ws_pk.data)  
  
# Evaluate Model ----------------------------------------------------------
  
  # ~ Predict test data -------------------------------------------------------

  # predict test data
  pred <- model %>% predict(unlist(data_test_x))
  
  # change predictions below 0 to 0
  pred <- ifelse(pred < 0, 0, pred)
  
  tt.id <- pTe.id[-test.tune.id]
    #which(!pTe.id %in% test.tune.id)
  
  # Eliminate frames which may just be noise
  pred <- ifelse(data_orig$fz[tt.id+lookback] < 50, NA, pred)
  
  # change flight to 0
  waist.z <- ifelse(imus == 1, 44, ifelse(imus == 2, 22, NA))
  pred <- ifelse(data_test_x[,lookback,waist.z] == min(data_test_x[,lookback,waist.z], na.rm = T), 0, pred)
  
  
  # create df of predictions and target
  plot.data <- data.frame(
    x = c(rep(NA, lookback), 1:length(data_test_tar)),
    prediction = c(rep(NA, lookback), (pred)), #[,12,1],
    target = c(rep(NA, lookback), data_test_tar),
    set = data_orig$set[tt.id],
    movement = data_orig$movement[tt.id],
    set_move = data_orig$set_move[tt.id],
    limb = data_orig$limb[tt.id],
    vgrf = data_orig$fz[tt.id],
    waist.acc = data.test$WAIST_r.acc[data.test$tune == 0][1:length(tt.id)],
    shank.acc = data.test$SHANK_r.acc[data.test$tune == 0][1:length(tt.id)]
    ) %>% 
    mutate(
      waist.tran = waist.acc  * waist.mod$coefficients[2] + waist.mod$coefficients [1],
      shank.tran = shank.acc  * shank.mod$coefficients[2] + shank.mod$coefficients [1],
      fz = data_orig$fz[tt.id]
    ) %>% 
    filter(vgrf != 0)
  
  
  # ~ Reverse scaling and normalization -------------------------------------
  
  # create function to reverse normalization of data
  inv.normalize <- function(norm, x.min, x.max){
    norm*(x.max-x.min)+x.min
  }
  
  if(target == 1){ref <- data.ref$achilles}
  if(target == 2){ref <- data.ref$patellartendon}
  if(target == 3){ref <- data.ref$tibia}
  if(target == 4){ref <- data.ref$grf}
  
  ref.min <- min(ref, na.rm = T)
  ref.max <- max(ref, na.rm = T)

  plot.data[,c(2, 3, 9, 10, 11, 12)] <- inv.normalize(plot.data[,c(2, 3, 9, 10, 11, 12)], ref.min, ref.max)
  
  # reverse scaling
  if(target == 1){ref <- which(names(mean) == "achilles")}
  if(target == 2){ref <- which(names(mean) == "patellartendon")}
  if(target == 3){ref <- which(names(mean) == "tibia")}
  if(target == 4){ref <- which(names(mean) == "grf")}
  
  mn <- mean[ref]
  sd <- std[ref]
  plot.data[,c(2, 3, 9, 10, 11, 12)] <- t((t(plot.data[,c(2, 3, 9, 10, 11, 12)]) * sd) + mn)
  
  #write to csv
  file.name <- paste0("output/pred_tar-", target, "_imus-", imus, "p", t, "TEST.csv") ###
  write.csv(plot.data, file.name)
  
  # save to global env
  assign(paste0("plot.data.", t), plot.data)
  
  # ~ Create peak data --------------------------------------------------------
  data_gc <- plot.data %>% 
    filter(
      !is.na(prediction),
      !is.na(target),
      !is.na(set),
      target > 0.5,
      !grepl("na", set_move, ignore.case = T)
    ) %>% 
    group_by(set_move, limb) %>% 
    summarise(
      target = max(target, na.rm = T),
      prediction = max(prediction, na.rm = T),
      waist = max(waist.acc, na.rm = T) * waist_pk.mod$coefficients[2] + waist_pk.mod$coefficients [1], #add coefficients
      shank = max(shank.acc, na.rm = T) * shank_pk.mod$coefficients[2] + shank_pk.mod$coefficients [1],
    ) %>% 
    arrange(target)

  #write to csv
  file.name <- paste0("output/pred_pk_tar-", target, "_imus-", imus, "p", t, "TEST.csv") ###
  write.csv(data_gc, file.name)
  
  # save to global env
  assign(paste0("data_gc.", t), plot.data)
  
}


# Evaluate models ---------------------------------------------------------

# create a long df of all outputs

for(i in c(1:4)){ # targets
  target <- i
  
  for(j in 1:2){ # imus
    imus <- j
    
    for(k in 1:6){ # participants
      
      file.path <- paste0("output/pred_tar-", target, "_imus-", imus, "p", k, ".csv")
      
      tissue <- case_when(
        target == 1 ~ "ach",
        target == 2 ~ "pat",
        target == 3 ~ "tib",
        target == 4 ~ "grf"
      )
      
      df_predictions <- read.csv(file.path) %>% 
        rename(
          freebody = target
        ) %>% 
        mutate(
          target_tissue = tissue,
          imus = imus,
          participant = k
        ) %>% 
        filter(
          !is.na(movement),
          !is.na(prediction),
          !is.na(freebody)
        )
      
      if(i == 1 & j == 1 & k == 1){df.eval <- df_predictions}else{
        df.eval <- rbind(df.eval, df_predictions)
      }
      
    }
  }
}

gcs <- unique(paste(df.eval$participant, df.eval$set_move))

df.eval.pk <- df.eval %>% 
  group_by(target_tissue, imus, participant, set_move, set, movement, limb) %>% 
  summarise(
    freebody = max(freebody, na.rm = T),
    prediction = max(prediction, na.rm = T),
    vgrf = max(vgrf, na.rm = T),
    waist.acc = max(waist.acc, na.rm = T),
    waist.tran = max(waist.tran, na.rm = T),
    shank.acc = max(shank.acc, na.rm = T),
    shank.tran = max(shank.tran, na.rm = T),
    error = abs(freebody - prediction),
    mape = error / freebody
    #waist = max(waist.acc, na.rm = T) * waist_pk.mod$coefficients[2] + waist_pk.mod$coefficients [1], #add 
    #shank = max(shank.acc, na.rm = T) * shank_pk.mod$coefficients[2] + shank_pk.mod$coefficients [1]
  )

results <- data.frame(
  imus = NA, target = NA, participant = NA, 
  r_nn = NA, r_waist = NA, r_shank = NA, 
  r.pk_nn = NA, r.pk_waist = NA, r.pk_shank = NA,
  mae_nn = NA, mae_waist = NA, mae_shank = NA, 
  rmse_nn = NA, rmse_waist = NA, rmse_shank = NA,
  mae.pk_nn = NA, mae.pk_waist = NA, mae.pk_shank = NA, 
  rmse.pk_nn = NA, rmse.pk_waist = NA, rmse.pk_shank = NA
)[-1,]

n <- 1

for(i in c(1:4)){ # targets
  target <- i
  
  for(j in 1:2){ # imus
    imus <- j
    
    for(k in 1:6){ # participants
      
      tissue <- case_when(
        target == 1 ~ "ach",
        target == 2 ~ "pat",
        target == 3 ~ "tib",
        target == 4 ~ "grf"
      )
      
      df.eval.pk.subset <- subset(df.eval.pk, target_tissue == tissue & imus == j & participant == k)
      df.eval.subset <- subset(df.eval, target_tissue == tissue & imus == j & participant == k)
      
      results[nrow(results)+1,] <- NA
      
      results$participant[n] <- k
      results$imus[n] <- j
      results$target[n] <- tissue
      
      results$r_nn[n] <-    cor(df.eval.subset$freebody, df.eval.subset$prediction, use = "complete.obs")
      results$r_waist[n] <- cor(df.eval.subset$freebody, df.eval.subset$waist.tran, use = "complete.obs")
      results$r_shank[n] <- cor(df.eval.subset$freebody, df.eval.subset$shank.tran, use = "complete.obs")
      
      results$r.pk_nn[n] <-    cor(df.eval.pk.subset$freebody, df.eval.pk.subset$prediction)
      results$r.pk_waist[n] <- cor(df.eval.pk.subset$freebody, df.eval.pk.subset$waist.tran)
      results$r.pk_shank[n] <- cor(df.eval.pk.subset$freebody, df.eval.pk.subset$shank.tran)
      
      results$mae_nn[n] <-    mean(abs(df.eval.subset$freebody - df.eval.subset$prediction), na.rm = T)
      results$mae_waist[n] <- mean(abs(df.eval.subset$freebody - df.eval.subset$waist.tran), na.rm = T)
      results$mae_shank[n] <- mean(abs(df.eval.subset$freebody - df.eval.subset$shank.tran), na.rm = T)
      
      results$rmse_nn[n] <-    sqrt(mean((df.eval.subset$freebody - df.eval.subset$prediction)^2, na.rm = T))
      results$rmse_waist[n] <- sqrt(mean((df.eval.subset$freebody - df.eval.subset$waist.tran)^2, na.rm = T))
      results$rmse_shank[n] <- sqrt(mean((df.eval.subset$freebody - df.eval.subset$shank.tran)^2, na.rm = T))
      
      results$mae.pk_nn[n] <-    mean(abs(df.eval.pk.subset$freebody - df.eval.pk.subset$prediction), na.rm = T)
      results$mae.pk_waist[n] <- mean(abs(df.eval.pk.subset$freebody - df.eval.pk.subset$waist.tran), na.rm = T)
      results$mae.pk_shank[n] <- mean(abs(df.eval.pk.subset$freebody - df.eval.pk.subset$shank.tran), na.rm = T)
      
      results$rmse.pk_nn[n] <-    sqrt(mean((df.eval.pk.subset$freebody - df.eval.pk.subset$prediction)^2, na.rm = T))
      results$rmse.pk_waist[n] <- sqrt(mean((df.eval.pk.subset$freebody - df.eval.pk.subset$waist.tran)^2, na.rm = T))
      results$rmse.pk_shank[n] <- sqrt(mean((df.eval.pk.subset$freebody - df.eval.pk.subset$shank.tran)^2, na.rm = T))
      
      n <- n + 1
    }
  }
}

results.summary <- results %>% 
  group_by(imus, target) %>% 
  summarise(
    r_nn =    mean(r_nn),
    r_waist = mean(r_waist),
    r_shank = mean(r_shank),
    #
    r.pk_nn =    mean(r.pk_nn),
    r.pk_waist = mean(r.pk_waist),
    r.pk_shank = mean(r.pk_shank),
    #
    r2_nn =    mean(r_nn)^2,
    r2_waist = mean(r_waist)^2,
    r2_shank = mean(r_shank)^2,
    #
    r2.pk_nn =    mean(r.pk_nn)^2,
    r2.pk_waist = mean(r.pk_waist)^2,
    r2.pk_shank = mean(r.pk_shank)^2,
    #
    mae_nn =    mean(mae_nn),
    mae_waist = mean(mae_waist),
    mae_shank = mean(mae_shank),
    #
    rmse_nn =    mean(rmse_nn),
    rmse_waist = mean(rmse_waist),
    rmse_shank = mean(rmse_shank),
    #
    mae.pk_nn =    mean(mae.pk_nn),
    mae.pk_waist = mean(mae.pk_waist),
    mae.pk_shank = mean(mae.pk_shank),
    #
    rmse.pk_nn =    mean(rmse.pk_nn),
    rmse.pk_waist = mean(rmse.pk_waist),
    rmse.pk_shank = mean(rmse.pk_shank)
  )

abs.res <- results.summary %>% 
  group_by(imus) %>% 
  summarise(
    mean = 
    across(r_nn:rmse.pk_shank, mean)
    
  )


# TABLES  ----------------------------------------------

table.1 <- results %>% 
  select(
    imus,
    target,
    participant,
    contains("rmse"),
  ) %>%   
  pivot_longer(
    contains("_"),
    names_sep = "_",
    names_to = c("variable", "algorithm")
  ) %>% 
  pivot_wider(
    names_from = variable,
    values_from = value
  ) %>% 
  pivot_wider(
    names_from = target,
    values_from = rmse:rmse.pk
  ) %>% 
  pivot_wider(
    names_from = algorithm,
    values_from = rmse_ach:rmse.pk_grf
  ) %>%
  pivot_wider(
    names_from = imus,
    values_from = rmse_ach_nn:rmse.pk_grf_shank
  ) %>% 
  arrange(participant) %>% 
  select(-c(contains("waist_2"), contains("shank_2"))) %>% 
  write.csv("tables/Table1.csv")

# Wrangle results data for plotting ---------------------------------------

df <- results %>% 
  pivot_longer(
    contains("_"),
    names_to = c("variable", "algorithm"),
    names_sep = "_"
  ) %>% 
  separate(
    "variable",
    c("variable", "type"),
    sep = "\\."
  ) %>% 
  replace_na(
    list(
      type = "all"
    )
  ) %>% 
  mutate(
    imus = ifelse(imus == 1, 5, 2),
    algorithm = paste0(algorithm, imus)
  ) %>% 
  filter(
    !algorithm %in% c("shank2", "waist2")
  ) %>% 
  mutate(
    target = paste0(type, target),
    target = factor(target, levels = c("allach", "allpat", "alltib", "allgrf", "pkach", "pkpat", "pktib", "pkgrf"),
                    labels = c("Achilles Tendon Force", "Patellar Tendon Force", "Tibia Force", "Ground Reaction Force",
                               "Peak Achilles Tendon Force", "Peak Patellar Tendon Force", "Peak Tibia Force", "Peak GRF")),
    algorithm = factor(algorithm, levels = c("nn5", "nn2", "shank5", "waist5"), labels = c("NN-5", "NN-2", "Shank", "Waist"))
  )

df.sum <- df %>% 
  group_by(
    algorithm, target, type, imus, variable
  ) %>% 
  summarise(
    value = mean(value)
  )

# Plot Results ------------------------------------------------------------

  plotly::plot_ly(data = df.eval[df.eval$participant == 1 & df.eval$imus == 2 & df.eval$target_tissue == "grf",], type = "scatter", mode = "lines") %>% 
    plotly::add_trace(x = ~x, y = ~freebody , name = "target", line = list(color = 'black'), opacity = 0.5) %>% 
    plotly::add_trace(x = ~x, y = ~prediction , name = "prediction", line = list(color = 'red'), opacity = 0.5, text = ~set_move)

# r -----------------------------------------------------------------------

r.pks <- ggplot(subset(df, type == "pk" & variable == "r"))+
  geom_point(
    data = subset(df.sum, type == "pk" & variable == "r"),
    aes(
      x = algorithm,
      y = value^2,
      col = algorithm, 
    ),
    fill = "white", shape = 23, size = 3
  )+
  scale_color_manual(values = c("red", "#283f3b", "blue", "#C0930A"))+
  geom_point(
    aes(
      x = algorithm,
      y = value^2,
      fill = algorithm
    ), 
    alpha = 0.3, shape = 22, colour = "transparent"
  )+
  scale_fill_manual(values = c("red", "#283f3b", "blue", "#C0930A"))+
  facet_wrap(~target, nrow = 1)+
  theme_minimal()+
  scale_y_continuous(name = expression(paste("Variation explained by prediction (", r^2, ")")),
                     limits = c(0, 1), breaks = round(seq(0, 1, 0.1)*10)/10, expand = c(0,0))+
  labs(
    x = NULL
  )+
  theme(
    axis.line = element_line(),
    axis.ticks = element_line(),
    panel.grid = element_blank(),
    panel.grid.major.y = element_line(colour = "#e6e6e6", size = 0.4),
    #panel.grid.minor.y = element_line(colour = "#e6e6e6", size = 0.4),
    panel.spacing.x = unit(5, "mm"),
    legend.position = "none",
    text = element_text(size = 10, hjust = 0.5, vjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text.x = element_text(angle = 35, hjust = 1)
    #strip.text = element_blank()
  )+
  ggsave(
    "C:/Users/shaw_/Documents/R/IMU/plots/peaks_r.png", 
    type = "cairo",
    height = 4, width = 18, units = "in"
  )

r.all <- ggplot(subset(df, type == "all" & variable == "r"))+
  #geom_segment(
  #  #data = df[1,],
  #  x = 0, xend = 4,
  #  y = 0, yend = 0,
  #  alpha = 0.01
  #)+
  geom_point(
    data = subset(df.sum, type == "all" & variable == "r"),
    aes(
      x = algorithm,
      y = value^2,
      col = algorithm, 
    ),
    fill = "white", shape = 23, size = 3
  )+
  scale_color_manual(values = c("red", "#283f3b", "blue", "#C0930A"))+
  geom_point(
    aes(
      x = algorithm,
      y = value^2,
      fill = algorithm
    ), 
    alpha = 0.3, shape = 22, colour = "transparent"
  )+
  scale_fill_manual(values = c("red", "#283f3b", "blue", "#C0930A"))+
  facet_wrap(~target, nrow = 1)+
  theme_minimal()+
  scale_y_continuous(name = expression(paste("Variation explained by prediction (", r^2, ")")),
                     limits = c(0, 1), breaks = round(seq(0, 1, 0.1)*10)/10, expand = c(0,0))+
  labs(
    x = NULL
  )+
  theme(
    axis.line = element_line(),
    axis.ticks = element_line(),
    panel.grid = element_blank(),
    panel.grid.major.y = element_line(colour = "#e6e6e6", size = 0.4),
    #panel.grid.minor.y = element_line(colour = "#e6e6e6", size = 0.4),
    panel.spacing.x = unit(5, "mm"),
    legend.position = "none",
    text = element_text(size = 10, hjust = 0.5, vjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text.x = element_text(angle = 35, hjust = 1)
    #axis.text.x = element_blank()
  )+
  ggsave(
    "C:/Users/shaw_/Documents/R/IMU/plots/all_r.png", 
    type = "cairo",
    height = 4, width = 18, units = "in"
  )

r.all / r.pks +
  theme(
    plot.margin = margin(4,0,4,0, "mm")
  )+
  ggsave(
    "C:/Users/shaw_/Documents/R/IMU/plots/r2_panels.png", 
    type = "cairo",
    height = 180, width = 170, units = "mm"
  )
  
# SCATTER PLOTS --------------------------------------------------------

# labeller
participants <- list(
  '1'="Participant 1",
  '2'="Participant 2",
  '3'="Participant 3",
  '4'="Participant 4",
  '5'="Participant 5",
  '6'="Participant 6"
)

part_labs <- function(variable,value){
  return(participants[value])
}

# Scatter - All Data ------------------------------------------------------

df.eval.pk.ach.2 <- subset(df.eval.pk, target_tissue == "ach" & imus == 2)
nn2.r <- rmcorr::rmcorr(participant = factor(df.eval.pk.ach.2$participant), 
                        df.eval.pk.ach.2$prediction, df.eval.pk.ach.2$freebody, dataset = df.eval.pk.ach.2)
sh.r <- rmcorr::rmcorr(participant = factor(df.eval.pk.ach.2$participant), 
                       df.eval.pk.ach.2$shank.tran, df.eval.pk.ach.2$freebody, dataset = df.eval.pk.ach.2)



ggplot(df.eval.pk.ach.2)+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 1
  )+
  theme_classic()

ggplot(df.eval.pk.ach.2)+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 1
  )+
  theme_classic()

ggplot(subset(df.eval.pk, imus == 2))+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction,
      group = participant
    ),
    col = "red", size = 0.1, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "IMU Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~target_tissue, nrow = 1)+ #, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )+
  ggsave("plots/nn2panels.png", type = "cairo", height = 60, width = 220, units = "mm")

ggplot(subset(df.eval.pk, imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.4,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran,
      group = participant
    ),
    col = "red", size = 0.1, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Shank Linear Regression\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~target_tissue, nrow = 1)+ #, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )+
  ggsave("plots/shank_panels.png", type = "cairo", height = 60, width = 220, units = "mm")

# Neural Network - All -----------------------------------

# ~ Achilles --------------------------------------------------------------


scatter.ach <- ggplot(subset(df.eval, target_tissue == "ach" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "NN-2 Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Tibia  ----------------------------------------------------------------

scatter.tib <- ggplot(subset(df.eval, target_tissue == "tib" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "NN-2 Tibial Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Patellar Tendon -------------------------------------------------------

scatter.pat <- ggplot(subset(df.eval, target_tissue == "pat" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 12.05))+
  scale_y_continuous(name = "NN-2 Pat. Tendon Force\nEstimate (BWs)", limits = c(-0.05, 12.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ GRF -------------------------------------------------------------------

scatter.grf <- ggplot(subset(df.eval, target_tissue == "grf" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.12, y = 0, yend = 16.12,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = "Laboratory Force Estimate (BWs)", limits = c(-0.02, 5.02))+
  scale_y_continuous(name = "NN-2 GRF\nEstimate (BWs)", limits = c(-0.02, 5.02))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    panel.grid.minor = element_blank(),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )

scatter.ach / scatter.pat / scatter.tib / scatter.grf+
  ggsave("plots/nn_scatter_all.png", width = 19, height = 14, units = "cm", type = "cairo")


# Neural Network - Peak -----------------------------------

# ~ Achilles --------------------------------------------------------------

scatter.ach.pk <- ggplot(subset(df.eval.pk, target_tissue == "ach" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "NN-2 Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Tibia  ----------------------------------------------------------------

scatter.tib.pk <- ggplot(subset(df.eval.pk, target_tissue == "tib" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "NN-2 Tibial Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Patellar Tendon -------------------------------------------------------

scatter.pat.pk <- ggplot(subset(df.eval.pk, target_tissue == "pat" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 12.05))+
  scale_y_continuous(name = "NN-2 Pat. Tendon Force\nEstimate (BWs)", limits = c(-0.05, 12.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ GRF -------------------------------------------------------------------

scatter.grf.pk <- ggplot(subset(df.eval.pk, target_tissue == "grf" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = prediction
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.12, y = 0, yend = 16.12,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = prediction
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = "Laboratory Force Estimate (BWs)", limits = c(-0.02, 5.02))+
  scale_y_continuous(name = "NN-2 GRF\nEstimate (BWs)", limits = c(-0.02, 5.02))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    panel.grid.minor = element_blank(),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )

# combine and save
scatter.ach.pk / scatter.pat.pk / scatter.tib.pk / scatter.grf.pk +
  ggsave("plots/nn_scatter_pk.png", width = 19, height = 14, units = "cm", type = "cairo")


# Shank - All -----------------------------------

# ~ Achilles --------------------------------------------------------------

scatter.ach.sh <- ggplot(subset(df.eval, target_tissue == "ach" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Shank Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Tibia  ----------------------------------------------------------------

scatter.tib.sh <- ggplot(subset(df.eval, target_tissue == "tib" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Shank Tibial Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Patellar Tendon -------------------------------------------------------

scatter.pat.sh <- ggplot(subset(df.eval, target_tissue == "pat" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 12.05))+
  scale_y_continuous(name = "Shank Pat. Tendon Force\nEstimate (BWs)", limits = c(-0.05, 12.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ GRF -------------------------------------------------------------------

scatter.grf.sh <- ggplot(subset(df.eval, target_tissue == "grf" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.12, y = 0, yend = 16.12,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = "Laboratory Force Estimate (BWs)", limits = c(-0.02, 5.02))+
  scale_y_continuous(name = "Shank GRF\nEstimate (BWs)", limits = c(-0.02, 5.02))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    panel.grid.minor = element_blank(),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )

scatter.ach.sh / scatter.pat.sh / scatter.tib.sh / scatter.grf.sh +
  ggsave("plots/sh_scatter_all.png", width = 19, height = 14, units = "cm", type = "cairo")



# Shank - Peak -----------------------------------

# ~ Achilles --------------------------------------------------------------

scatter.ach.pk.sh <- ggplot(subset(df.eval.pk, target_tissue == "ach" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Shank Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Tibia  ----------------------------------------------------------------

scatter.tib.pk.sh <- ggplot(subset(df.eval.pk, target_tissue == "tib" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Shank Tibial Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Patellar Tendon -------------------------------------------------------

scatter.pat.pk.sh <- ggplot(subset(df.eval.pk, target_tissue == "pat" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 12.05))+
  scale_y_continuous(name = "Shank Pat. Tendon Force\nEstimate (BWs)", limits = c(-0.05, 12.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ GRF -------------------------------------------------------------------

scatter.grf.pk.sh <- ggplot(subset(df.eval.pk, target_tissue == "grf" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = shank.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.12, y = 0, yend = 16.12,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = shank.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = "Laboratory Force Estimate (BWs)", limits = c(-0.02, 5.02))+
  scale_y_continuous(name = "Shank GRF\nEstimate (BWs)", limits = c(-0.02, 5.02))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    panel.grid.minor = element_blank(),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )

# combine and save
scatter.ach.pk.sh / scatter.pat.pk.sh / scatter.tib.pk.sh / scatter.grf.pk.sh +
  ggsave("plots/sh_scatter_pk.png", width = 19, height = 14, units = "cm", type = "cairo")

# Waist - All -----------------------------------

# ~ Achilles --------------------------------------------------------------

scatter.ach.wa <- ggplot(subset(df.eval, target_tissue == "ach" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Waist Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Tibia  ----------------------------------------------------------------

scatter.tib.wa <- ggplot(subset(df.eval, target_tissue == "tib" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Waist Tibial Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Patellar Tendon -------------------------------------------------------

scatter.pat.wa <- ggplot(subset(df.eval, target_tissue == "pat" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 12.05))+
  scale_y_continuous(name = "Waist Pat. Tendon Force\nEstimate (BWs)", limits = c(-0.05, 12.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ GRF -------------------------------------------------------------------

scatter.grf.wa <- ggplot(subset(df.eval, target_tissue == "grf" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.05,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.12, y = 0, yend = 16.12,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = "Laboratory Force Estimate (BWs)", limits = c(-0.02, 5.02))+
  scale_y_continuous(name = "Waist GRF\nEstimate (BWs)", limits = c(-0.02, 5.02))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    panel.grid.minor = element_blank(),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )

scatter.ach.wa / scatter.pat.wa / scatter.tib.wa / scatter.grf.wa +
  ggsave("plots/wa_scatter_all.png", width = 19, height = 14, units = "cm", type = "cairo")



# Waist - Peak -----------------------------------

# ~ Achilles --------------------------------------------------------------

scatter.ach.pk.wa <- ggplot(subset(df.eval.pk, target_tissue == "ach" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Waist Achilles Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1, labeller = part_labs)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Tibia  ----------------------------------------------------------------

scatter.tib.pk.wa <- ggplot(subset(df.eval.pk, target_tissue == "tib" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 16.05))+
  scale_y_continuous(name = "Waist Tibial Force\nEstimate (BWs)", limits = c(-0.05, 16.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ Patellar Tendon -------------------------------------------------------

scatter.pat.pk.wa <- ggplot(subset(df.eval.pk, target_tissue == "pat" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.05, y = 0, yend = 16.05,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = NULL, limits = c(-0.05, 12.05))+
  scale_y_continuous(name = "Waist Pat. Tendon Force\nEstimate (BWs)", limits = c(-0.05, 12.05))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )


# ~ GRF -------------------------------------------------------------------

scatter.grf.pk.wa <- ggplot(subset(df.eval.pk, target_tissue == "grf" & imus == 2))+
  geom_point(
    aes(
      x = freebody,
      y = waist.tran
    ),
    alpha = 0.5,
    size = 0.2,
    shape = 16,
    colour = "black"
  )+
  geom_segment(
    x = 0, xend = 16.12, y = 0, yend = 16.12,
    col = "blue", size = 0.1
  )+
  geom_smooth(
    aes(
      x = freebody,
      y = waist.tran
    ),
    col = "red", size = 0.3, method = "lm", se = F
  )+
  scale_x_continuous(name = "Laboratory Force Estimate (BWs)", limits = c(-0.02, 5.02))+
  scale_y_continuous(name = "Waist GRF\nEstimate (BWs)", limits = c(-0.02, 5.02))+
  coord_cartesian(expand = F)+
  facet_wrap(~participant, nrow = 1)+
  theme_minimal()+
  theme(
    legend.position = "none",
    panel.spacing.x = unit(4, "mm"),
    panel.border = element_rect(fill = "transparent", size = 0.5),
    panel.grid.minor = element_blank(),
    axis.ticks = element_line(size = 0.2),
    strip.text.x = element_blank(),
    text = element_text(size = 7),
    panel.grid = element_blank()
  )

# combine and save
scatter.ach.pk.wa / scatter.pat.pk.wa / scatter.tib.pk.wa / scatter.grf.pk.wa +
  ggsave("plots/wa_scatter_pk.png", width = 19, height = 14, units = "cm", type = "cairo")
