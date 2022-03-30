#Synthetic Control Chart Creator
#Organize input, with variable input depending on the type of anomaly pattern 

#Basic Pattern Creation
normal<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1))
}

#Equation: y(t)=mu+r(t)*sigma+a*sin(2*pi*t/T)
#r(t): inevitable accidental fluctuation with Gaussian distribution
#a: amplitude of cyclic pattern
#T: period of cyclic pattern
cyclic<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)+runif(numobs,1.5*sigma,2.5*sigma)*sin(2*seq(1,numobs,by=1)*pi/16))
}

#Equation: y(t)=mu+r(t)*sigma+d*(-1)^t
#d: degree of state departure
systematic<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)+runif(numobs,1*sigma,3*sigma)*(-1)^seq(1,numobs,by=1))
}

#Equation: y(t)=mu+r(t)*sigma*[0.2;0.4]
stratification<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)*runif(numobs,0.2*sigma,0.4*sigma))
}

#Equation: y(t)=mu+r(t)*sigma+t*g
#g: gradient of upward trend
ut<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)+seq(1,numobs,by=1)*runif(numobs,0.05*sigma,0.25*sigma))
  }
#Equation: y(t)=mu+r(t)*sigma-t*g
#g: gradient of downward trend
dt<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)-seq(1,numobs,by=1)*runif(numobs,0.05*sigma,0.25*sigma))
}

#Equation: y(t)=mu+r(t)*sigma+k*s
#k=1 if t>=P, else=0
#P: moment of shift
#s: amplitude of shift
us<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)+(runif(numobs,10,20)<=seq(1,numobs,by=1))*runif(numobs,1*sigma,3*sigma))
}

#Equation: y(t)=mu+r(t)*sigma-k*s
#k=1 if t>=P, else=0
#P: moment of shift
#s: amplitude of shift
ds<-function(mu,sigma,numobs){
  return(mu+sigma*rnorm(numobs,0,1)-(runif(numobs,10,20)<=seq(1,numobs,by=1))*runif(numobs,1*sigma,3*sigma))
}

#Joining Normal + Anomaly at a given point
DataCreation<-function(mu,sigma,anstart,anend,typeanomaly){
  if ( typeanomaly=="cyclic") {
    return(c(normal(mu,sigma,anstart),cyclic(mu,sigma,anend-anstart)))
  } else if ( typeanomaly=="systematic") {
      return(c(normal(mu,sigma,anstart),systematic(mu,sigma,anend-anstart)))
  } else if ( typeanomaly=="stratification") {
      return(c(normal(mu,sigma,anstart),stratification(mu,sigma,anend-anstart)))
  } else if ( typeanomaly=="ut") {
      return(c(normal(mu,sigma,anstart),ut(mu,sigma,anend-anstart)))
  } else if ( typeanomaly=="dt") {
      return(c(normal(mu,sigma,anstart),dt(mu,sigma,anend-anstart)))
  } else if ( typeanomaly=="us") {
      return(c(normal(mu,sigma,anstart),us(mu,sigma,anend-anstart)))
  } else if ( typeanomaly=="ds") {
      return(c(normal(mu,sigma,anstart),ds(mu,sigma,anend-anstart)))
  }
}

#Creating in-control data set to extract statistical features and create model
TrainingSetCreation<-function(mu,sigma,numobs,size){
  i<-1
  df <- data.frame(normal(mu,sigma,numobs))
  
  for (i in seq(2,size,by=1)){
    df<- cbind(df, normal(mu,sigma,numobs))
  }
  colnames(df)<-paste("col", 1:size, sep = "")
  return(df)
}


