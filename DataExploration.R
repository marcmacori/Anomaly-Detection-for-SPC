#Import data
SC <- read.table("C:/Users/Marc/Desktop/TFG/Data/synthetic_control.data", quote="\"", comment.char="")
#Classify data
Classification<-rep(c("Normal","Cyclic","IT","DT","US","DS"),each=100)
SCClass <- cbind(SC,Classification)
