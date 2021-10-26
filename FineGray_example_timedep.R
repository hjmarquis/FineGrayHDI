
# Install my glmnet type of package for PH models with LTRC entries (from-to outcome)
install.packages("cmprskHD_1.0.0.tar.gz", repos = NULL, type = "source")

#============================================
# Example with initial Lasso through my phnet 
#============================================

library(cmprskHD)
library(survival)
library(parallel)
source("one_step_FineGray_timedep.R")
source("FineGray_timedep.R")
source("FineGray_gen.R")

n 					= 400
p 					= 30

# Baseline probability of event from cause 1
prob=0.5
# set true beta values
non_zero_coeff 		= c(0.5,0.5)
beta1 				= c(non_zero_coeff, rep(0, (p - length(non_zero_coeff))))	
beta2=rep(c(-.5,.5),p/2)

# Generate data
dat=data.gen.FG(n,prob,beta1,beta2)
Z=as.matrix(dat[,grepl('^Z',names(dat))])
cause=dat$cause
Y= Y.phnet = dat$surv
Y.phnet[cause==2,2] = 0

# Get the initial beta from phnet (surprisingly fast)
cv.lasso = cv.phnet(Z,Y, fstatus = cause, family="finegray")
init.beta = drop(coef(cv.lasso$phnet.fit, s=cv.lasso$lambda.min))
init.beta[1:5]

# Fit Fine-Gray
fit=main(Z,Y,cause,init_beta_ = init.beta)

# Initial estimator
fit$initBeta[1:5]
# One-step estimator
fit$bhat[1:5]
# Estimated Standard Error
fit$se.LASSO[1:5]
# 95 % Confidance interval
fit$bhat[1:5] + 1.96 * fit$se.LASSO[1:5] # upper
fit$bhat[1:5] - 1.96 * fit$se.LASSO[1:5] # lower


#============================================
# Example with time-dependent covariates
# only possible through my phnet 
#============================================

library(cmprskHD)
library(survival)
library(parallel)
source("one_step_FineGray_timedep.R")
source("FineGray_timedep.R")
source("FineGray_gen.R")

n 					= 400
p 					= 30

# Baseline probability of event from cause 1
prob=0.5
# set true beta values
# for similicity of simulation design, 
# the relevant covariates are not changing with time
non_zero_coeff 		= c(0.5,0.5)
beta1 				= c(non_zero_coeff, rep(0, (p - length(non_zero_coeff))))	
beta2=rep(c(-.5,.5,0),c(2,2,p-4))

# Generate data
# Each id associates with multiple from-to entries for time-dependent covariates
# The coding of survival entries is tricky, so the data generation steps are shown here.
#----------------------------------------------------------------------------------------
Z1=cov.gen.stdN(n,length(beta1)) # Covaraite at baseline
Z2=cov.gen.stdN(n,length(beta1)) # Covariate after the change
Z2[,1:4] = Z1[,1:4] # keep the relevant covriate
phaz=drop(exp(Z1%*%beta1)) # relative subdistribution risk cause 1
phaz2 = drop(exp(Z1%*%beta2)) # cause 2 parameter

cause=1+rbinom(n,1,prob=(1-prob)^phaz)

# Generate the final sojourn times
ftime=rep(0,n)
ftime[cause==1]=FG.invT1(phaz[cause==1],prob)
ftime[cause==2]=rT2.exp(phaz2[cause==2])

Ctime=rcensor.unif(n) # Censoring time
cause[Ctime<ftime] = 0 
ftime = pmin(ftime,Ctime)
cov.change = runif(n,0,1) # Time of covariate change

# Prepare data for phnet
#----------------------------

# Most entries follow conventional coding
cause01 = which(cause<2)
n01 = length(cause01)
end.before.change = (ftime <= cov.change)[cause01]
fstatus1 = end.before.change*cause[cause01]
surv1 = Surv(rep(0,n01),pmin(cov.change,ftime)[cause01],fstatus1)
sec.entries = cause01[!end.before.change]
surv2 = Surv(cov.change[sec.entries],ftime[sec.entries],cause[sec.entries])
fstatus2 = cause[sec.entries]

# Cause 2 events are different!
cause2 = which(cause == 2)
n2 = length(cause2)
e2.after.change = (ftime >= cov.change)[cause2]
entries.before.type2 = cause2[e2.after.change]
n2.0 = length(entries.before.type2)
surv3 = Surv(rep(0,n2.0),
             cov.change[entries.before.type2],
             rep(0,n2.0))
fstatus3 = rep(0,n2.0)
e2.before.change = cause2[!e2.after.change]
n2.2 = n2-n2.0
surv4 = Surv(c(rep(0,n2.2),cov.change[cause2]),
             c(cov.change[e2.before.change], 
               rep(max(ftime[cause==1])+1,n2)),
             rep(0,n2.2+n2))
fstatus4 = rep(2,n2.2+n2)

# Put everthing together
id = c(cause01,sec.entries,entries.before.type2,
       e2.before.change,cause2)
Y = rbind(surv1,surv2,surv3,surv4)
Z = rbind(Z1[cause01,],Z2[sec.entries,],
          Z1[entries.before.type2,],Z1[e2.before.change,],
          Z2[cause2,])
ftime = ftime[id]
fstatus = c(fstatus1,fstatus2,fstatus3,fstatus4)

# Get the initial beta from phnet (surprisingly fast)
lasso = phnet(Z,Y, ftime = ftime, fstatus = fstatus,
                 id=id,family="finegray")
cv.lasso = cv.phnet(Z,Y, ftime = ftime, fstatus = fstatus,
                    id=id,family="finegray")
init.beta = drop(coef(cv.lasso$phnet.fit, s=cv.lasso$lambda.min))

init.beta[1:5]

# Fit Fine-Gray using the init.beta
id.order = order(id)
Z = Z[id.order,]
Y = Y[id.order,]
Y = Surv(Y[,1],Y[,2],Y[,3])
fstatus = fstatus[id.order]
ftime = ftime[id.order]
id = id[id.order]
fit=main(Z,Y,fstatus,ftime=ftime,id=id,init_beta_ = init.beta)

# Initial estimator
fit$initBeta[1:5]
# One-step estimator
fit$bhat[1:5]
# Estimated Standard Error
fit$se.LASSO[1:5]
# 95 % Confidance interval
fit$bhat[1:5] + 1.96 * fit$se.LASSO[1:5] # upper
fit$bhat[1:5] - 1.96 * fit$se.LASSO[1:5] # lower


