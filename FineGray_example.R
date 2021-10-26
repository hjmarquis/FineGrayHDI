library(glmnet)
library(survival)
library(parallel)
source("one_step_FineGray.R")
source("FineGray.R")
source("FineGray_gen.R")

n 					= 100
p 					= 30

# Baseline probability of event from cause 1
prob=0.3
# set true beta values
non_zero_coeff 		= c(0.5,0.5)
beta1 				= c(non_zero_coeff, rep(0, (p - length(non_zero_coeff))))	
beta2=rep(c(-.5,.5),p/2)

# Generate data
  dat=data.gen.FG(n,prob,beta1,beta2)
  Y=dat$surv
  Z=as.matrix(dat[,grepl('^Z',names(dat))])
  cause=dat$cause

# Fit Fine-Gray
  fit=main(Z,Y,cause)

# Initial estimator
  fit$initBeta[1:5]
# One-step estimator
  fit$bhat[1:5]
# Estimated Standard Error
  fit$se.LASSO[1:5]
# 95 % Confidance interval
  fit$bhat[1:5] + 1.96 * fit$se.LASSO[1:5] # upper
  fit$bhat[1:5] - 1.96 * fit$se.LASSO[1:5] # lower
