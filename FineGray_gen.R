#require(survival)

# ---------------------------------------------------------------------------
# Simulation Generators
# ---------------------------------------------------------------------------

# Cumulative incidence function at t
# phaz= exp(beta*Z), p: baseline probability of P(T < +infinity)
CIF1.FG=function(t,phaz,p)
{
  return(1-(1-p*(1-exp(-t)))^phaz)
}

# Generate T_1 from cumulative incidence function
# with proportional hazard
FG.invT1=function(phaz,p)
{
  U=runif(length(phaz))
  
  return(-log(1-(1-(1-U*(1-(1-p)^phaz))^(1/phaz))/p ))
}

# Generate n by p covaraites, std Normal
cov.gen.stdN=function(n,p)
{
  return(matrix(rnorm(n*p),n,p))
}

# Generate n by p covaraites, 4 blocks Schumacher
cov.gen.Binder=function(n,p)
{
  cor1=matrix(0.5,4,4)+diag(rep(0.5,4))
  block1=mvrnorm(n,rep(0,4),cor1)
  
  cor2=matrix(0.35,4,4)+diag(rep(0.65,4))
  block2=mvrnorm(n,rep(0,4),cor2)
  
  cor3=matrix(0.05,8,8)+diag(rep(0.95,8))
  block3=mvrnorm(n,rep(0,8),cor3)
  
  block4=matrix(rnorm(n*(p-16)),n,p-16)
  
  return(cbind(block1,block2,block3,block4))
}

# Generate censoring times, Uniform(2,5)
rcensor.unif=function(n,a=2,b=5)
{
  return(runif(n,a,b))
}

# Generate event times from other causes
# phaz=exp(beta*Z)
rT2.exp=function(phaz)
{
  return(rexp(length(phaz),rate = phaz))
}

# Generate simulation data
data.gen.FG=function(n,prob,beta1,beta2,cov.gen=cov.gen.stdN,rT1=FG.invT1,rT2=rT2.exp,rcensor=rcensor.unif)
{
  Z=cov.gen(n,length(beta1))
  
  phaz=exp(Z%*%beta1)
  
  cause=1+rbinom(n,1,prob=(1-prob)^phaz)
  
  T=rep(0,n)
  T[cause==1]=rT1(phaz[cause==1],prob)
  T[cause==2]=rT2(exp(Z[cause==2,]%*%beta2))
  
  C=rcensor(n)
  
  surv=Surv(pmin(T,C),T<=C)
  cause=cause*surv[,2]
  T[cause==2] = Inf
  
  return(data.frame(surv=surv,cause=cause, CC.time = pmin(T,C),Z=Z))
}