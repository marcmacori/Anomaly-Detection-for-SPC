# Load class package
library(class)
library(caret)
#Import data
SC <- read.table("C:/Users/Marc/Desktop/TFG/Data/synthetic_control.data", quote="\"", comment.char="")
SC <- transpose(SC)

#Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x))) }
NormSCC <- as.data.frame(lapply(SC, normalize))
NormSCC <- transpose(NormSCC)

#Classify data
Classification<-rep(c("Normal","Anomaly"),times=c(100,500))
SCClass <- cbind(NormSCC,Classification)

#Slicing Data
set.seed(123)
dat.d <- sample(1:nrow(SCClass),size=nrow(SCClass)*0.7,replace = FALSE) #random selection of 70% data.

train.SCC <- SCClass[dat.d,1:60] # 70% training data
test.SCC <- SCClass[-dat.d,1:60] # remaining 30% test data

#Creating seperate dataframe for 'Creditability' feature which is our target.
train.SCC_Cat<- SCClass[dat.d,61]
test.SCC_Cat <-SCClass[-dat.d,61]

#Find the number of observation
NROW(train.SCC_Cat) 

#KNN Model
knn.20 <- knn(train=train.SCC, test=test.SCC, cl=train.SCC_Cat, k=20)
knn.21 <- knn(train=train.SCC, test=test.SCC, cl=train.SCC_Cat, k=21)

#Calculate the proportion of correct classification for k = 20, 21
ACC.20 <- 100 * sum(test.SCC_Cat == knn.20)/NROW(test.SCC_Cat)
ACC.21 <- 100 * sum(test.SCC_Cat == knn.21)/NROW(test.SCC_Cat)

confusionMatrix(table(knn.20 ,test.SCC_Cat))
