# Build neural networks for the estimation of physiological tissue
# forces from inertial measurement unit data using all participant data

# Written by Joseph Shaw (shaw_joseph@outlook.com)

# select target (Achilles tendon = 1, patellar tendon = 2, tibia = 3, GRF = 4)
target <- 1

# select imus (All = 1, Waist/Shank = 2)
imus <- 2

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
  
  # assign to participant reference in global env
  assign(i, data)
  
}

# bind each participant's data
data_orig <- rbind(p01, p02, p03, p04, p05, p06)

# make vector of anthropometry column indexes
col.ids <- which(multigrep::multigrep(c("com", "mass", "long", "sag", "tran"), names(data_orig)))

# drop columns
data <- data_orig %>% 
  select(-col.ids) %>%   # remove anthro columns
  select(-c(1:7))        # remove index columns

# make a vector of targets for easy removal later
targets <- which(names(data) %in% c("achilles", "anklex", "ankley", "anklez", 
                                    "patellartendon", "fx", "fy", "fz", "grf", "tibia")
                 )

# divide into train / test / validate
data.train <- data
ws.data_orig <- data_orig

# Scale data --------------------------------------------------------------

# create function to normalize data
normalize <- function(x, y) {
  return ((x - min(y, na.rm = T)) / (max(y, na.rm = T) - min(y, na.rm = T)))
}

# scale
mean <- apply(data.train, 2, mean, na.rm = T)
std <- apply(data.train, 2, sd, na.rm = T)
min <- apply(data.train, 2, min, na.rm = T)
max <- apply(data.train, 2, max, na.rm = T)


data.train <- as.data.frame(scale(data.train, center = mean, scale = std)) #scaled data frames

# copy of data train for test and val normalization
data.ref <- data.train

min <- apply(data.train, 2, min, na.rm = T)
max <- apply(data.train, 2, max, na.rm = T)

# normalize each column based on training data
for(i in 1:ncol(data.train)){
  data.train[,i] <- normalize(data.train[,i], y = data.ref[,i])
}

values <- as.data.frame(t(data.frame(
  mean = mean,
  std = std,
  min = min,
  max = max
)))

values_in <- values  %>% 
  select(
    contains("SHANK"),
    contains("WAIST")
  )

values_out <- values  %>% 
  select(
    achilles, patellartendon, tibia, grf
  )

values <- cbind(values[,1:11], values[,1:11], values[,12:22])

write.csv(values_in, paste(target, "values.csv"), row.names = F, col.names = F)
write.csv(values_out, paste(target, "values_out.csv"), row.names = F, col.names = F)

# re-add limb data
data.train$limb <- data_orig$limb-1 

# Create time windows -----------------------------------------------------

ws.data <- data.train
ws.data$set_move <- data_orig$set_move

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
dat <- data.train
max_index <- nrow(data.train)
data_rnn_ts <- create.ts(step = step, dat = dat, lookback = lookback, delay = delay, 
                         min_index = 1, max_index = max_index)
data_rnn <- data_rnn_ts()



# Bind time windows -------------------------------------------------------

# ~training
# bind windows
data_train_x <-   data_rnn[[1]]
data_train_ach <- data_rnn[[2]]
data_train_tib <- data_rnn[[3]]
data_train_pat <- data_rnn[[4]]
data_train_grf <- data_rnn[[5]]

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


# select target
if(target == 1){data_train_tar <- data_train_ach}
if(target == 2){data_train_tar <- data_train_pat}
if(target == 3){data_train_tar <- data_train_tib}
if(target == 4){data_train_tar <- data_train_grf}

# divide training into training and validation

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
}
}

if(!is.null(drop)){
  data_val_x <- data_val_x[,,-c(drop)]
  data_val_x_r <- data_val_x_r[,,-c(drop)]
  data_train_x_r <- data_train_x_r[,,-c(drop)]
}

# shuffle data
rows <- sample(nrow(data_train_x_r), replace = F)

data_train_x_r <- data_train_x_r[rows, ,]
data_train_tar_r <- data_train_tar_r[rows]





# Model 1: Achilles Force | All IMUs --------------------------------------

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
  
  save_model_hdf5(model, "achilles_5.h5")
  
  model %>% save_model_tf("achilles_5")
  
}


# Model 2: Achilles Force | Waist & Shank ---------------------------------

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
  
  
  save_model_hdf5(model, "achilles_2.h5")
  
  model %>% save_model_tf("achilles_2")
}
# Model 3: PT Force | All IMUs --------------------------------------------

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
  
  save_model_hdf5(model, "pat_tendon_5.h5")
  
  model %>% save_model_tf("pat_tendon_5")
}


# Model 4: PT Force | Waist & Shank ---------------------------------------

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
  
  
  save_model_hdf5(model, "pat_tendon_2.h5")
  
  model %>% save_model_tf("pat_tendon_2")
}

# Model 5: Tibial Force | All IMUs ---------------------------------------

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
  
  save_model_hdf5(model, "tibia_5.h5")
  
  model %>% save_model_tf("tibia_5")
  
}  

# Model 6: Tibial Force | Waist & Shank ---------------------------------------

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
  
  save_model_hdf5(model, "tibia_2.h5")
  
  model %>% save_model_tf("tibia_2")
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
  
  save_model_hdf5(model, "grf_5.h5")
  
  model %>% save_model_tf("grf_5")
  
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
  
  save_model_hdf5(model, "grf_2.h5")
  
  model %>% save_model_tf("grf_2")
}


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
