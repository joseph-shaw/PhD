# Script containing code to create a function to identify jumps from accelerometer data

# Arguments ----------------------------------------------------------------------------

# file - String indicating the filepath
# data.freq - Numeric indicating the sampling frequency of the accelerometer
# order - Numeric indicating the filter order
# filt.freq - Numeric indicating the filter cut-off frequency

# Load Packages ------------------------------------------------------------------------

library(tidyverse)

# Create data processing function ------------------------------------------------------

process_imu_data <- function(file, data.freq, order, filt.freq){
  
  # create filter
  nyquist.freq <- data.freq / 2
  bf <- signal::butter(order, filt.freq / nyquist.freq, type = "low")
  
  # Read data --------------------------------------------------------------------------
  df <- fread(file) %>%
    rename(time = 1,  x.acc = 2,  y.acc = 3,  z.acc = 4) %>%  
    mutate(
      x.raw = x.acc,
      y.raw = y.acc,
      z.raw = z.acc
    ) %>%
    mutate_at(2:4, signal::filtfilt, filt = bf) %>% 
    mutate_at(2:4, as.vector) %>%
    mutate(
      res.acc = sqrt(x.acc^2 + y.acc^2 + z.acc^2),
      up.id = ifelse(z.acc < 0.2 | z.raw < 0.2, 0, z.acc) # flatten flight 
    )
  
  df2 <- df
  df2[1,1] <- 10000
  
  # Iterate flight smoothing
  while(min(df == df2, na.rm = TRUE) == 0){
    df2 <- df
    
    df$up.id  <- ifelse(
      runmin(df$up.id, 10 ,align = "left") == 0 &
        runmin(df$up.id, 10, align = "right") == 0 & 
        df$up.id < 0.6, 
      0, df$up.id)
    
    df$up.id <- ifelse(
      runmin(df$up.id, 5 ,align = "left") == 0 &
        runmin(df$up.id, 5 ,align = "right") == 0 &
        df$up.id < 1.5, 
      0, df$up.id)
  }
  
  # Additional smoothing for extreme whip
  df$up.id <- ifelse(
    runmin(df$up.id, 6 ,align = "left") == 0 &
      runmin(df$up.id, 6 ,align = "right") == 0 & 
      df$up.id < 5, 
    0, df$up.id)
  
  df$up.id <- ifelse(
    runmin(df$up.id, 20 ,align = "left") == 0 & 
      runmin(df$up.id, 20 ,align = "right") == 0 & 
      runmax(df$up.id, 30 ,align = "center") < 1,
    0, df$up.id)
  
  while(min(df == df2, na.rm = TRUE) == 0){
    df2 <- df
    
    df$up.id  <- ifelse(
      runmin(df$up.id, 10 ,align = "left") == 0 & 
        runmin(df$up.id, 10, align = "right") == 0 & 
        df$up.id < 0.6, 
      0, df$up.id)
    
    df$up.id <- ifelse(
      runmin(df$up.id, 5 ,align = "left") == 0 &
        runmin(df$up.id, 5 ,align = "right") == 0 &
        df$up.id < 1.5, 
      0, df$up.id)
    
    df$up.id <- ifelse(
      runmin(df$up.id, 20 ,align = "left") == 0 & 
        runmin(df$up.id, 20 ,align = "right") == 0 & 
        runmax(df$up.id, 30 ,align = "center") < 1,
      0, df$up.id)
  }
  
  # If up acc is 0, make resultant 0
  df$resultant.id <- ifelse(
    df$up.id == 0, 0, sqrt(df$up.id ^2 + df$x.acc ^2 + df$y.acc ^2)
    )
  
  #Identify peaks
  df$peak <- ifelse(
    df$resultant.id == runmax(df$resultant.id, 45, align = "center") & 
    df$resultant.id >= 1.65,
    1, 0)
  
  #Calculate the magnitude of the peak
  df$peak.mag <- ifelse(df$peak == 1, df$resultant.id, 0)
  df$peak.up.mag <- ifelse(df$peak == 1, df$z.acc, 0)
  
  df[is.na(df)] <- 0
  
  #Identify periods of flight
  df$flight <- ifelse(df$up.id == 0, 1, 0)
  
  #Make row number helper column
  df$row <- 1:nrow(df)
  
  #Identify onset and end of jump, and peaks
  df$flight.s.f <- ifelse(df$flight == 1 & lag(df$flight, 1) == 0, 'A',
                      ifelse(df$flight == 0 & lag(df$flight, 1) == 1, 'B',
                         ifelse(df$peak == 1 & df$peak.up.mag > 1.35, 'C','')
                         )
                      )
  
  df$flight.s.f <- ifelse(lag(df$flight.s.f,1) == 'B' & lag(df$peak,1) == 1 & 
                            lag(df$peak.up.mag,1) > 1.35, 
                          'C', df$flight.s.f
                          )
  
  #Calculate duration of jump
  as <- which(df$flight.s.f == "A")
  bs <- which(df$flight.s.f == "B")
  cs <- which(df$flight.s.f == "C")
  
  #time to jump landing
  next_b <- base::sapply(as, function(a) {
    diff <- bs-a
    if(all(diff < 0)) return(NA)
    bs[min(diff[diff > 0]) == diff]
  })
  
  df$f.time <- NA
  df$f.time[as] <- df$row[next_b]
  df$f.time <- df$f.time - df$row
  
  #Calculate time to next peak
  next_c <- base::sapply(as, function(a) {
    diff <- cs-a
    if(all(diff < 0)) return(NA)
    cs[min(diff[diff > 0]) == diff]
  })
  df$ttp <- NA
  df$ttp[as] <- df$row[next_c]
  df$ttp <- df$ttp - df$row
  
  df$l.to.p <- ifelse(df$f.time > 0, df$ttp - df$f.time, 0)
  
  #calculate landing acceleration
  next_c <- base::sapply(as, function(a) {
    diff <- cs-a
    if(all(diff < 0)) return(NA)
    cs[min(diff[diff > 0]) == diff]
  })
  df$landing.acc <- NA
  df$landing.acc[as] <- df$peak.mag[next_c]
  
  #Eliminate flight times beyond 0.8 s and below 0.15 s
  df$f.time <- ifelse(df$f.time > 80, 0,
                      ifelse(df$f.time < 15, 0, df$f.time))
  
  #Check if there's takeoff peak present
  df$takeoff.peak <- ifelse(
    df$flight.s.f == 'A' &
      runmax(df$peak.mag, 40, align = "right", endrule = "NA") > 1.65 &
      df$f.time > 1, 
    1, 0)
  
  #Is it a jump?
  df$Jump <- ifelse(df$flight.s.f == 'A' &
                      df$f.time > 22 &
                      df$f.time < 80 &
                      df$l.to.p < 38 &
                      df$takeoff.peak == 1, 1, 0)  
  
  df <- df %>% 
    mutate(
      #If a jump is identified, calculate the height?
      jh = ifelse(Jump == 1, 9.81 * (f.time/10)^2 / 8, NA),
      jh = (0.9301*jh) - 0.9443
    )
  
  return(df)
}