#Western Electric Rules Implementation

if (!require("weco")) {
  install.packages("weco")
  library("weco")
}

#Using weco package on dataset1

TimeSeries1 <- read.csv(file = "TimeSeries1.csv",
                        header = TRUE, row.names = 1)
TS1 <- as.data.frame(t(TimeSeries1))
TS1_Class <- as.data.frame(t(TimeSeries1_Classification))

WE_AD <- function(data, numTS, numnorm){
  a <- list()
  for (i in 1:numTS) {
    b <- weco.combine(data[,i],  sdx = sd(data[1:numnorm, i]), mux = mean(data[1:numnorm, i]),
                      lst.rules = list(list(1), list(5), list(6), list(8)))
    a[[i]] <- b$weco
  }
  return(a)
}

WE_TS1_Class <- as.data.frame(WE_AD(TS1, length(TS1), 20))
colnames(WE_TS1_Class) <- 1:length(WE_TS1_Class)

write.csv(WE_TS1_Class,"TimeSeries1_WE_Classification.csv")

#using weco on dataset2

TimeSeries2 <- read.csv(file = "TimeSeries2.csv",
                        header = TRUE, row.names = 1)
WE_TS2_Class <- as.data.frame(WE_AD(TimeSeries2, 5, 1000))
colnames(WE_TS2_Class) <- 1:length(WE_TS2_Class)

write.csv(WE_TS2_Class,"TimeSeries2_WE_Classification.csv")