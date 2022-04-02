#Extract statistical features from a data set
StatExtract<-function(data){
  if(!require("dplyr")) {
    install.packages("dplyr")
    library("dplyr")
  }
  if(!require("moments")) {
    install.packages("moments")
    library("moments")
  }
  
  M<-transmute(rowwise(df), M=mean(c_across(cols=everything())))
  SD<-transmute(rowwise(df), SD=sd(c_across(cols=everything())))
  KURT<-transmute(rowwise(df),KURT=kurtosis(c_across(cols=everything())))
  SKEW<-transmute(rowwise(df),SKEW=skewness(c_across(cols=everything())))
  return(Sfeatures<-cbind(M,SD,KURT,SKEW))
}