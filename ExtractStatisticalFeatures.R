#Extract statistical features from a data set
stat_extract <- function(data) {
  if (!require("dplyr")) {
    install.packages("dplyr")
    library("dplyr")
  }
  if (!require("moments")) {
    install.packages("moments")
    library("moments")
  }
  m <- transmute(rowwise(df), M = mean(c_across(cols = everything())))
  sd <- transmute(rowwise(df), SD = sd(c_across(cols = everything())))
  kurt <- transmute(rowwise(df), KURT = kurtosis(c_across(cols = everything())))
  skew <- transmute(rowwise(df), SKEW = skewness(c_across(cols = everything())))
  return(sfeatures <- cbind(m, sd, kurt, skew))
}