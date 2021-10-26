#source("FineGray_gen.R")
#source("FineGray.R")
#require(glmnet)
#require(doParallel)
#library('parallel')
#library('scalreg')

# ---------------------------------------------------------------------------
# parallel computing
# ---------------------------------------------------------------------------

# any positive real number a: the used lambda will be a * lambda_{cv}


main	 			= function(Z, Y , cause, ftime=NULL, id = 1:nrow(Z), init.tuning.method = 'cv',para.tuning.method = 1, ncores = 4,
                    summary.col = 1:ncol(Z),
                    init_beta_ = NA){		
  
  p = ncol(Z)
  #  n_event_ 		= sum(Y[, 2])
  n_sum_ 			= length(unique(id))
  
  FG.dat=FG.Surv(Y,cause,ftime, Z, id)
  
  # no package can solve Cox LASSO with truncation (related to time-dependent covariates)
  # except our "cmprskHD" to be published
  
  if(all(is.na(init_beta_)))
  {
    init.Y = Surv(FG.dat$surv[,2], FG.dat$surv[,3])
    
    init_fit_ 		= init.est(Z = FG.dat$Z, Y = init.Y, IPW = FG.dat$weight)
    if(init.tuning.method == 'cv'){
      init_beta_      = init_fit_$beta.cv	
    } else if(init.tuning.method == '1se'){
      init_beta_      = init_fit_$beta.1se			
    } else if(init.tuning.method == '1se.small'){
      init_beta_      = init_fit_$beta.1se.small			
    } else {
      stop('Oops')
    }
  }
  

  
  tmp.LASSO = FG.score(Z = FG.dat$Z, Y = FG.dat$surv ,IPW = FG.dat$weight,
                       X = FG.dat$X, id = FG.dat$id, C.surv = FG.dat$C.surv, beta.est= init_beta_)
  
  out = list()
  out$initBeta = init_beta_
  
  temp.LASSO.p = score.nodewiselasso(x = tmp.LASSO$hat_Z_, ncores = ncores, lambdaseq = 'quantile', 
                                     nsum = n_sum_, lambdatuningfactor = para.tuning.method,
                                     summary.col = 1:p)
  out$bhat = init_beta_ + temp.LASSO.p$out %*% tmp.LASSO$S_
  var.LASSO = 1/n_sum_ * temp.LASSO.p$out %*% tmp.LASSO$curly_V_ %*% t(temp.LASSO.p$out)
  out$se.LASSO = sqrt(diag(var.LASSO))
  
  # tmp.onestep = FG.score(Z = FG.dat$Z, Y = FG.dat$surv ,IPW = FG.dat$weight,
  #                        X = FG.dat$X, id = FG.dat$id, C.surv = FG.dat$C.surv, beta.est= out$bhat)
  # temp.onestep = score.nodewiselasso(x = tmp.onestep$hat_Z_, ncores = ncores, lambdaseq = 'quantile', 
  #                                    nsum = n_sum_, lambdatuningfactor = para.tuning.method,
  #                                    summary.col = summary.col)
  # var.onestep = 1/n_sum_ * temp.onestep$out %*% tmp.onestep$curly_V_ %*% t(temp.onestep$out)
  # out$se.onestep  = sqrt(diag(var.onestep))
  
  return(out)
}



# ---------------------------------------------------------------------------
# auxiliary functions
# ---------------------------------------------------------------------------


# initial estimator: lasso estimator
# ---------------------------------------------------------------------------
init.est = function(Z, Y, IPW){

  temp=cv.glmnet(Z,Y,family="cox",weights=IPW)
  min.ind = which.min(temp$cvm)
  ind.1se.small = max(which(temp$cvlo <= temp$cvm[min.ind]))
  ind.1se = min(which(temp$cvm <= temp$cvup[min.ind]))
  beta.cv = temp$glmnet.fit$beta[, min.ind]
  beta.1se = temp$glmnet.fit$beta[, ind.1se]
  beta.1se.small = temp$glmnet.fit$beta[, ind.1se.small]
  
  return(list(beta.cv = beta.cv, beta.1se = beta.1se, beta.1se.small = beta.1se.small))
  
}


# final estimator
# ---------------------------------------------------------------------------
b.hat 		= function(init.est, theta.est, score.est){
  
  # temp_ 	= init.est - theta.est %*% t(score.est)
  temp_ 	= init.est + theta.est %*% t(score.est)
  return(temp_)
}



# ---------------------------------------------------------------------------
# scalreg functions
# ---------------------------------------------------------------------------

score.nodewiselasso <- function(x,
                                wantTheta = TRUE,
                                verbose = FALSE,
                                lambdaseq = "quantile",
                                ncores = 4,
                                parallel = TRUE,
                                oldschool = FALSE,
                                lambdatuningfactor = 1, nsum,
                                summary.col)
{
  ## Purpose:
  ## This function calculates the score vectors Z OR the matrix of nodewise
  ## regression coefficients Thetahat, depending on the argument wantTheta.
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  
  ## First, a sequence of lambda values over all nodewise regressions is created
  
  if(lambdaseq == "quantile"){ ## this is preferred
    lambdas <- nodewise.getlambdasequence(x,summary.col)
  }else{
    if(lambdaseq == "linear"){
      lambdas <- nodewise.getlambdasequence.old(x,verbose,summary.col)
    }else{
      stop("invalid lambdaseq given")
    }
  }
  
  if(verbose){
    print("Using the following lambda values")
    print(lambdas)
  }
  
  ## 10-fold cv is done over all nodewise regressions to calculate the error
  ## for the different lambda
  cvlambdas <- cv.nodewise.bestlambda(lambdas = lambdas, x = x,
                                      parallel = parallel, ncores = ncores,
                                      oldschool = oldschool,
                                      summary.col = summary.col
                                      )
  if(verbose){
    print(paste("lambda.min is", cvlambdas$lambda.min))
    print(paste("lambda.1se is", cvlambdas$lambda.1se))
  }
  
  if(lambdatuningfactor == "lambda.1se"){
    if(verbose)
      print("lambda.1se used for nodewise tuning")
    ## We use lambda.1se for bestlambda now!!!
    bestlambda <- cvlambdas$lambda.1se
  }else if(lambdatuningfactor == 'lambda.1se.small'){
    bestlambda = cvlambdas$lambda.1se.small
  } else {
    if(verbose)
      print(paste("lambdatuningfactor used is ", lambdatuningfactor, sep = ""))
    
    bestlambda <- cvlambdas$lambda.min.list * lambdatuningfactor
  }
  
  if(verbose){
    print("Picked the best lambda")
    print(bestlambda)
    ##print("with the error ")
    ##print(min(err))
  }
  
  ## Having picked the 'best lambda', we now generate the final Z or Thetahat
  if(wantTheta){
    out <- score.getThetaforlambda(x = x,
                                   lambda = bestlambda,
                                   parallel = parallel,
                                   ncores = ncores,
                                   oldschool = TRUE,
                                   verbose = verbose, nsum = nsum,
                                   summary.col = summary.col)
  }else{
    Z <- score.getZforlambda(x = x, lambda = bestlambda, parallel = parallel,
                             ncores = ncores, oldschool = oldschool,
                             summary.col = summary.col)
    out <- Z
  }
  return.out <- list(out=out,
                     bestlambda = bestlambda)
  return(return.out)
}

score.getThetaforlambda <- function(x, lambda, parallel = FALSE, ncores = 8,
                                    oldschool = FALSE, verbose = FALSE,
                                    oldtausq = FALSE, nsum,summary.col)
{
  ## Purpose:
  ## This function is for calculating Thetahat once the desired tuning
  ## parameter is known
  ## ----------------------------------------------------------------------
  ## Arguments:qq
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  # print("Calculating Thetahat by doing nodewise regressions and dropping the unpenalized intercept")
  n <- nrow(x)
  p <- ncol(x)
  m <- length(summary.col)
  C <- diag(rep(1,p))[,summary.col]
  T2 <- numeric(m)
  
  if(oldschool){
    # print("doing getThetaforlambda oldschool")
    for(j in 1:m){
      i = summary.col[j]
      # browser()
      glmnetfit <- glmnet(x[,-i], x[,i], intercept = F)
      coeffs <- as.vector(predict(glmnetfit,x[,-i], type = "coefficients",
                                  s = lambda[j]))[-1]
      ## we just leave out the intercept
      
      C[-i,j] <- -as.vector(coeffs)
      if(oldtausq){
        ## possibly quite incorrect,it ignores the intercept, but intercept is
        ## small anyways. Initial simulations show little difference.
        T2[j] <- as.numeric(crossprod(x[,i]) / n - x[,i] %*% (x[,-i] %*%
                                                                coeffs) / n)
      }else{   
        ##print("now doing the new way of calculating tau^2")
        T2[j] <- as.numeric((x[,i] %*%
                               (x[,i] - predict(glmnetfit,x[,-i],s =lambda[j])))/nsum)
      }
    }
  }else{
    stop("not implemented yet!")
  }
  thetahat <- C %*% solve(diag(T2))
  if(verbose){
    print("1/tau_j^2")
    print(solve(diag(T2)))
  }
  ##this is thetahat ^ T!!
 thetahat <- t(thetahat)

 if(all(thetahat[lower.tri(thetahat)] == 0,
        thetahat[upper.tri(thetahat)] == 0) && verbose)
   print("Thetahat is a diagonal matrix!!!! ")
  
  return(thetahat)
}




cv.nodewise.bestlambda <- function(lambdas, x, K = 10, parallel = FALSE,
                                   ncores = 8, oldschool = FALSE,
                                   summary.col)
{
  ## Purpose:
  ## this function finds the optimal tuning parameter value for minimizing
  ## the K-fold cv error of the nodewise regressions.
  ## A second value of the tuning parameter, always bigger or equal to the
  ## former, is returned which is calculated by allowing the cv error to
  ## increase by the amount of
  ## 1 standard error (a similar concept as to what is done in cv.glmnet).
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  
  n <- nrow(x)
  p <- ncol(x)
  l <- length(lambdas)
  m <- length(summary.col)
  
  ## Based on code from cv.glmnet for sampling the data
  dataselects <- sample(rep(1:K, length = n))
  
  if(oldschool){
    # print("doing cv.nodewise.error oldschool")
    totalerr <- numeric(l)
    for(j in 1:m){ ## loop over the nodewise regressions
      c = summary.col[j]
      for(i in 1:K){ ## loop over the test sets
        whichj <- dataselects == i ## the test part of the data
        
        glmnetfit <- glmnet(x[!whichj,-c,drop=FALSE], x[!whichj,c,drop=FALSE],
                            lambda=lambdas)
        predictions <- predict(glmnetfit,x[whichj, -c, drop = FALSE],
                               s = lambdas)
        totalerr <- totalerr + apply((x[whichj,c]-predictions)^2, 2, mean)
      }
    }
    totalerr <- totalerr / (K * m)
    stop("lambda.1se not implemented for oldschool cv.nodewise.bestlamba")
  }else{
    ## REPLACING THE FOR LOOP
    
    ##totalerr <- matrix(nrow = l, ncol = p)
    
    if(parallel){
      totalerr <- mcmapply(cv.nodewise.err.unitfunction,
                           c = summary.col,
                           K = K,
                           dataselects = list(dataselects = dataselects),
                           x = list(x = x),
                           lambdas = list(lambdas = lambdas),
                           mc.cores = ifelse(Sys.info()[1]=="Windows",1,ncores),
                           SIMPLIFY = FALSE)
    }else{
      totalerr <- mapply(cv.nodewise.err.unitfunction,
                         c = summary.col,
                         K = K,
                         dataselects = list(dataselects = dataselects),
                         x = list(x = x),
                         lambdas = list(lambdas = lambdas),
                         SIMPLIFY = FALSE)
    }
    # browser()
    ## Convert into suitable array (lambda, cv-fold, predictor)
    err.array  <- array(unlist(totalerr), dim = c(length(lambdas), K, m))
    err.mean   <- apply(err.array, 1, mean) ## 1 mean for each lambda
    
    ## for every lambda x predictor combination, get a mean
    err.mean.lambda.predictor = apply(err.array, c(1, 3), mean)
    ## for every lambda x predictor combination, get a standard error
    err.se.lambda.predictor = apply(err.array, c(1, 3), sd) / sqrt(K)
    
    pos.min.predictor = apply(err.mean.lambda.predictor, 2, function(x){which.min(x)})
    value.min.predictor = apply(err.mean.lambda.predictor, 2, function(x){min(x)})
    
    
    
    ## calculate mean for every lambda x fold combination (= average over p)
    ## for every lambda then get the standard errors (over folds)
    err.se     <- apply(apply(err.array, c(1, 2), mean), 1, sd) / sqrt(K)
    ##totalerr <- apply(totalerr, 1, mean)
  }
  
  pos.min    <- which.min(err.mean)
  lambda.min <- lambdas[pos.min]
  lambda.min.list = lambdas[pos.min.predictor]
  
  stderr.lambda.min <- err.se[pos.min]
  ##-   stderr.lambda.min <- cv.nodewise.stderr(K = K,
  ##-                                           x = x,
  ##-                                           dataselects = dataselects,
  ##-                                           lambda = lambda.min,
  ##-                                           parallel = parallel,
  ##-                                           ncores = ncores)
  lambda.1se <- max(lambdas[err.mean < (min(err.mean) + stderr.lambda.min)])
  lambda.1se.small <- min(lambdas[err.mean < (min(err.mean) + stderr.lambda.min)])
  # browser()
  return(list(lambda.min=lambda.min,
              lambda.1se=lambda.1se, lambda.1se.small = lambda.1se.small, lambda.min.list = lambda.min.list))
}

cv.nodewise.err.unitfunction <- function(c, K, dataselects, x, lambdas)
{
  ## Purpose:
  ## this method returns the K-fold cv error made by the nodewise regression
  ## of the single column c of x vs the other columns for all values of the
  ## tuning parameters in lambdas.
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  
  totalerr <- cv.nodewise.totalerr(c = c,
                                   K = K,
                                   dataselects = dataselects,
                                   x = x,
                                   lambdas = lambdas)
  ##return(apply(totalerr, 1, mean))
  return(totalerr)
}

cv.nodewise.totalerr <- function(c, K, dataselects, x, lambdas)
{
  ## Purpose:
  ## this method returns the error made for each fold of a K-fold cv
  ## of the nodewise regression of the single column c of x vs the other
  ## columns for all values of the tuning parameters in lambdas.
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  
  totalerr <- matrix(nrow = length(lambdas), ncol = K)
  
  for(i in 1:K){ ## loop over the test sets
    whichj <- dataselects == i ##the test part of the data
    
    glmnetfit <- glmnet(x = x[!whichj,-c, drop = FALSE],
                        y = x[!whichj, c, drop = FALSE],
                        lambda = lambdas)
    predictions  <- predict(glmnetfit, newx = x[whichj, -c, drop = FALSE],
                            s = lambdas)
    totalerr[, i] <- apply((x[whichj, c] - predictions)^2, 2, mean)
  }
  
  return(totalerr)
}

nodewise.getlambdasequence <- function(x,summary.col)
{
  ## Purpose:
  ## this method returns a lambdasequence for the nodewise regressions
  ## by looking at the automatically selected lambda sequences
  ## for each nodewise regression by glmnet.
  ## Equidistant quantiles of the complete set of lambda values are returned.
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  
  nlambda <- 100 ## use the same value as the glmnet default
  m <- length(summary.col)
  
  lambdas <- c()
  for(j in 1:m){
    c = summary.col[j]
    lambdas <- c(lambdas,glmnet(x[,-c],x[,c])$lambda)
  }
  
  lambdas <- quantile(lambdas, probs = seq(0, 1, length.out = nlambda))
  lambdas <- sort(lambdas, decreasing = TRUE)
  return(lambdas)
}

nodewise.getlambdasequence.old <- function(x,verbose=FALSE, summary.col)
{
  ## Purpose:
  ## this method returns a lambdasequence for the nodewise regressions
  ## by looking at the automatically selected lambda sequences
  ## for each nodewise regression by glmnet.
  ## It returns a __linear__ interpolation of lambda values between the max and
  ## min lambda value found.
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## ----------------------------------------------------------------------
  ## Author: Ruben Dezeure, Date: 27 Nov 2012 (initial version),
  nlambda <- 100#take the same value as glmnet does automatically
  m <- length(summary.col)
  maxlambda <- 0
  minlambda <- 100
  
  for(j in 1:m){
    c = summary.col[j]
    lambdas <- glmnet(x[,-c],x[,c])$lambda
    
    ##DEBUG
    if(verbose || is.nan(max(lambdas))){
      print(paste("c ",c,sep=""))
      print("lambdas")
      print(lambdas)
      print("max(lambdas) max(lambdas,na.rm=TRUE) maxlambda")
      print(paste(max(lambdas),max(lambdas,na.rm=TRUE),maxlambda,sep=""))
    }
    if(max(lambdas,na.rm=TRUE) > maxlambda){
      maxlambda <- max(lambdas,na.rm=TRUE)
    }
    if(min(lambdas,na.rm=TRUE) < minlambda){
      minlambda <- min(lambdas,na.rm=TRUE)
    }
  }
  
  lambdas <- seq(minlambda,maxlambda,by=(maxlambda-minlambda)/nlambda)
  lambdas <- sort(lambdas,decreasing=TRUE)
  return(lambdas)
}




# ---------------------------------------------------------------------------
# data generating
# ---------------------------------------------------------------------------
gen.func 	= function(n, p, beta0, Z, C, seed = 1){
  
  set.seed(seed)
  h_ 		= as.vector(exp(Z %*% beta0))
  
  #truelifetime
  
  X_ 		= rexp(n, h_)
  
  delta_  = ifelse(C >= X_, 1, 0)
  
  #event time
  Tt_ 	= ifelse(C >= X_, X_, C)
  
  Y_ 		= cbind(time = Tt_, status = delta_)
  
  return(Y_)
  
}  


# generating exponentially decay
ECcov = function(p, a){
  
  temp = matrix(a, ncol = p, nrow = p)
  diag(temp) = rep(1, p)
  return(temp)
  
}	

# data generating
# exp: lambda
# weibull: shape and scale
# gompertz: lambda and alpha
gen.func.more 	= function(n, p, beta0, Z, C, lifetime = c('exp', 'weibull', 'gompertz'), seed = 1, lambda = 1, alpha, shape, scale){
  
  set.seed(seed)
  h_ 		= as.vector(exp(Z %*% beta0))
  U_      = runif(n, 0, 1)
  
  #truelifetime
  if(lifetime == 'exp'){
    X_  = -log(U_)/(lambda * h_)
  }
  if(lifetime == 'weibull'){
    X_ 	= (-log(U_)/(scale * h_))^(1/shape)
  }
  if(lifetime == 'gompertz'){
    X_  = (1/alpha) * log(1 - (alpha * log(U_))/(lambda * h_))
  }
  
  delta_  = ifelse(C >= X_, 1, 0)
  
  #event time
  Tt_ 		= ifelse(C >= X_, X_, C)
  
  Y_ 		= cbind(time = Tt_, status = delta_)
  
  return(Y_)
  
}  


ECCov = function(p, rho){
  
  a = matrix(rho, nrow = p, ncol = p)
  diag(a) = rep(1, p)
  return(a)
}


# ---------------------------------------------------------------------------
# for my own use
# ---------------------------------------------------------------------------

censoring.rate = function(Y){
  return(1 - sum(Y[, 2])/length(Y[, 2]))
}


# Test the variance estimator

true.hessian = function(Z, Y, beta){
  
  n = dim(Z)[1]
  p = dim(Z)[2]
  n.event = sum(Y[, 2])
  ind.event = which(Y[, 2] == 1)
  
  temp = matrix(0, nrow = p, ncol = p)
  for(i in ind.event){
    
    # weighted average
    aa = weighted.average(t = Y[i, 1], Z = Z, Y = Y, beta.est = beta)$ave
    
    # normalised weights
    bb = weight(t = Y[i, 1], Z = Z, Y = Y, beta.est = beta)
    bb = bb/sum(bb)
    
    new.temp = matrix(0, nrow = p, ncol = p)
    for(i in 1: n){
      
      new.temp = tcrossprod(as.vector(Z[i, ] - aa)) * bb[i] + new.temp
    }
    temp = temp + new.temp
  }
  
  return(temp)	
}
