#' The main function to fit the Fine-Gray model with high-dimensional covariates
#'
#' \code{phnet}  fit a Proportional Hazards model with lasso or elasticnet.
#'
#' @param x input matrix, of dimension nobs x nvars; each row is an observation vector.
#' Can be in sparse matrix format (inherit from class "\code{sparseMatrix}"
#' as in package \code{Matrix})
#' @param y a matrix with columns named 'time' and 'status', or 'start', 'stop' and
#' 'status' for left-truncated data with time-dependent
#' covariates.
#' The last is a binary variable, with '1' indicating observed event (of the interested cause),
#' and '0' indicating right censored.
#' The function \code{Surv()} in package \code{survival} produces such a matrix.
#' @param ftime a vector of real values, the final time for each observations
#' under the Fine-Gray model with time-dependent covariates.
#' @param fstatus a vector of integer values, the type of cause for competing risks.
#' The cause of interest is coded as '1'.
#' Censoring is coded as '0'.
#' Other causes are coded as integers starting from '2'.
#' @param family Response type. Either "cox" or "finegray".
#' @param weights observation weights. Can be total counts if responses are proportion matrices.
#' Default is 1 for each observation
#' @param offset A vector of length nobs that is included in the linear predictor.
#' Default is NULL. If supplied, then values must also be supplied to the predict function.
#' @param alpha	The elasticnet mixing parameter, with \eqn{0\le \alpha \le 1}.
#' The penalty is defined as \deqn{(1-\alpha)/2\|\beta\|_2^2+\alpha\|\beta\|_1.}
#' \code{alpha=1} is the lasso penalty, and \code{alpha=0} the ridge penalty.
#' @param nlambda	The number of \code{lambda} values - default is 100.
#' @param lambda.min.ratio	Smallest value for \code{lambda}, as a fraction of \code{lambda.max},
#' the (data derived) entry value (i.e. the smallest value for which all coefficients are zero).
#' The default depends on the sample size \code{nobs} relative to the number of variables
#' \code{nvars}.
#' If \code{nobs > nvars}, the default is \code{0.0001}, close to zero. If \code{nobs < nvars},
#' the default is \code{0.01}.
#'  A very small value of lambda.min.ratio will lead to a saturated fit in the \code{nobs < nvars}
#'   case.
#' @param lambda	A user supplied \code{lambda} sequence.
#' Typical usage is to have the program compute its own \code{lambda} sequence based
#' on \code{nlambda} and \code{lambda.min.ratio}. Supplying a value of \code{lambda} overrides this.
#' WARNING: use with care. Avoid supplying a single value for \code{lambda}
#' (for predictions after CV use \code{predict()} instead).
#' Supply instead a decreasing sequence of \code{lambda} values.
#' phnet relies on its warm starts for speed, and its often faster
#' to fit a whole path than compute a single fit.
#' @param standardize	Logical flag for \code{x} variable standardization,
#' prior to fitting the model sequence.
#' The coefficients are always returned on the original scale.
#' Default is standardize=TRUE. If variables are in the same units already,
#' you might not wish to standardize.
#' @param thresh Convergence threshold for coordinate descent.
#' Each inner coordinate-descent loop continues until the maximum change in
#' the objective after any coefficient update is less than thresh times the null deviance.
#' Defaults value is \code{1E-7}.
#' @param dfmax Limit the maximum number of variables in the model.
#' Useful for very large \code{nvars}, if a partial path is desired.
#' @param pmax Limit the maximum number of variables ever to be \code{nonzero}.
#' @param exclude Indices of variables to be excluded from the model.
#' Default is none. Equivalent to an infinite penalty factor (next item).
#' @param penalty.factor Separate penalty factors can be applied to each coefficient.
#' This is a number that multiplies \code{lambda} to allow differential shrinkage.
#' Can be 0 for some variables, which implies no shrinkage,
#' and that variable is always included in the model.
#' Default is 1 for all variables (and implicitly infinity for variables listed in \code{exclude}).
#' Note: the penalty factors are internally rescaled to sum to nvars,
#' and the lambda sequence will reflect this change.
#' @param lower.limits Vector of lower limits for each coefficient;
#' default \code{-Inf}. Each of these must be non-positive.
#' Can be presented as a single value (which will then be replicated),
#' else a vector of length \code{nvars}.
#' @param upper.limits Vector of upper limits for each coefficient; default \code{Inf}.
#' See \code{lower.limits}.
#' @param maxit	Maximum number of passes over the data for all \code{lambda} values;
#' default is \code{10^5}.
#'
#' @details
#' \code{y} can have either 2 components for fixed covariates, sojourn time and status, or 3 components
#' for time-dependent covariates, entry time, exit time and status. The status in \code{y} is coded
#' as '1' for the final entry of observed events from the cause of interest and as '0' otherwise.
#'
#' With the time-dependent covariates, all entries exit prior to the final sojourn time
#' are marked as '0' for \code{fstatus}. Entries of a subject with a competing risks event
#' enter after the final sojourn time should be included in the data and are marked as
#' codes greater than '2'.
#' The \code{ftime} of these entries should be the associated competing risks event times.
#' The 'stop' in \code{y} for these entries should not mark the competing risks event times
#' but rather the covariate-change times.
#' Please code 'stop' in \code{y} for the last entry of a subject with the competing risks event
#' as the maximal event time.
#'
#' @return An object with \code{S3} class \code{"phnet"}.
#' \item{call}{the call that produced this object}
#' \item{beta}{a \code{nvars} by \code{length(lambda)} matrix of coefficients,
#' stored in sparse column format (\code{"CsparseMatrix"}). }
#' \item{lambda}{The actual sequence of \code{lambda} values used. When \code{alpha=0},
#' the largest \code{lambda} reported does not quite give the zero coefficients reported
#' (\code{lambda=inf} would in principle). Instead, the largest \code{lambda} for \code{alpha=0.001}
#'  is used, and the sequence of \code{lambda} values is derived from this.}
#' \item{dev.ratio}{The fraction of (null) deviance explained.
#' The deviance calculations incorporate weights if present in the model.
#' The deviance is defined to be 2*(loglike_sat - loglike),
#' where loglike_sat is the log-likelihood for the saturated model
#' (a model with a free parameter per observation). Hence dev.ratio=1-dev/nulldev.}
#' \item{nulldev}{Null deviance (per observation). This is defined to be
#' 2*(loglike_sat -loglike(Null)); The NULL model refers to the 0 model. }
#' \item{df}{The number of nonzero coefficients for each value of \code{lambda}.}
#' \item{dim}{dimension of coefficient matrix (ices).}
#' \item{nobs}{number of observations.}
#' \item{npasses}{total passes over the data summed over all \code{lambda} values.}
#' \item{offset}{a logical variable indicating whether an offset was included in the model.}
#' \item{jerr}{error flag, for warnings and errors (largely for internal debugging).}
#' @author Marquis (Jue) Hou
#' @export phnet
#' @seealso \code{\link[glmnet]{glmnet}}





phnet=function(x,y,ftime=NULL,fstatus=NULL,id=1:nrow(x),family=c("cox","finegray"),
                weights,offset=NULL,alpha=1.0,nlambda=100,
                lambda.min.ratio=ifelse(nobs<nvars,1e-2,1e-4),
                lambda=NULL,standardize=TRUE,thresh=1e-7,
                dfmax=nvars+1,pmax=min(dfmax*2+20,nvars),exclude,
                penalty.factor=rep(1,nvars),lower.limits=-Inf,
                upper.limits=Inf,maxit=100000)
{
  ### Prepare all the generic arguments, then hand off to family functions
  family=match.arg(family)
  if(alpha>1){
    warning("alpha >1; set to 1")
    alpha=1
  }
  if(alpha<0){
    warning("alpha<0; set to 0")
    alpha=0
  }
  alpha=as.double(alpha)

  this.call=match.call()
  nlam=as.integer(nlambda)
  y=drop(y) # we dont like matrix responses unless we need them
  np=dim(x)
  ###check dims
  if(is.null(np)|(np[2]<=1))stop("x should be a matrix with 2 or more columns")
  nobs=as.integer(np[1])
  if(missing(weights))weights=rep(1,nobs)
  else if(length(weights)!=nobs)stop(paste("number of elements in weights (",length(weights),") not equal to the number of rows of x (",nobs,")",sep=""))
  nvars=as.integer(np[2])
  dimy=dim(y)
  nrowy=ifelse(is.null(dimy),length(y),dimy[1])
  if(nrowy!=nobs)stop(paste("number of observations in y (",nrowy,") not equal to the number of rows of x (",nobs,")",sep=""))
  vnames=colnames(x)
  if(is.null(vnames))vnames=paste("V",seq(nvars),sep="")
  ne=as.integer(dfmax)
  nx=as.integer(pmax)
  if(missing(exclude))exclude=integer(0)
  if(any(penalty.factor==Inf)){
    exclude=c(exclude,seq(nvars)[penalty.factor==Inf])
    exclude=sort(unique(exclude))
  }
  if(length(exclude)>0){
    jd=match(exclude,seq(nvars),0)
    if(!all(jd>0))stop("Some excluded variables out of range")
    penalty.factor[jd]=1 #ow can change lambda sequence
    jd=as.integer(c(length(jd),jd))
  }else jd=as.integer(0)
  vp=as.double(penalty.factor)
  ###check on limits
  internal.parms=phnet.control()
  if(any(lower.limits>0)){stop("Lower limits should be non-positive")}
  if(any(upper.limits<0)){stop("Upper limits should be non-negative")}
  lower.limits[lower.limits==-Inf]=-internal.parms$big
  upper.limits[upper.limits==Inf]=internal.parms$big
  if(length(lower.limits)<nvars){
    if(length(lower.limits)==1)lower.limits=rep(lower.limits,nvars)else stop("Require length 1 or nvars lower.limits")
  }
  else lower.limits=lower.limits[seq(nvars)]
  if(length(upper.limits)<nvars){
    if(length(upper.limits)==1)upper.limits=rep(upper.limits,nvars)else stop("Require length 1 or nvars upper.limits")
  }
  else upper.limits=upper.limits[seq(nvars)]
  cl=rbind(lower.limits,upper.limits)
  if(any(cl==0)){
    ###Bounds of zero can mess with the lambda sequence and fdev; ie nothing happens and if fdev is not
    ###zero, the path can stop
    fdev=phnet.control()$fdev
    if(fdev!=0) {
      phnet.control(fdev=0)
      on.exit(phnet.control(fdev=fdev))
    }
  }
  storage.mode(cl)="double"
  ### end check on limits

  isd=as.integer(standardize)
  thresh=as.double(thresh)
  if(is.null(lambda)){
    if(lambda.min.ratio>=1)stop("lambda.min.ratio should be less than 1")
    flmin=as.double(lambda.min.ratio)
    ulam=double(1)
  }
  else{
    flmin=as.double(1)
    if(any(lambda<0))stop("lambdas should be non-negative")
    ulam=as.double(rev(sort(lambda)))
    nlam=as.integer(length(lambda))
  }
  is.sparse=FALSE
  ix=jx=NULL
  if(inherits(x,"sparseMatrix")){##Sparse case
    is.sparse=TRUE
    x=as(x,"CsparseMatrix")
    x=as(x,"dgCMatrix")
    ix=as.integer(x@p+1)
    jx=as.integer(x@i+1)
    x=as.double(x@x)
  }

  if(ncol(y)==3 && min(y[,1])==max(y[,1]))
  {
    y[,1]=y[,2]-y[,1]
    y=y[,-2]
    colnames(y)[1]="time"
  }

  is.tdep=(ncol(y)==3)
  ycr=cr=NULL
  if(family=="finegray")
  {
    if(is.null(fstatus))
      stop("Must provide final status (fstatus) for Fine-Gray model. ")
    cr = as.integer(fstatus)
    if(is.tdep)
    {
      if(is.null(ftime))
        stop("Must provide final time (ftime) for Fine-Gray model with
             time dependent covariates. ")
      ycr= as.double(ftime)
    }
  }

  fit=switch(family,
             "cox"=coxnet(x,is.sparse,is.tdep,ix,jx,y,weights,offset,alpha,nobs,
                          nvars,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,isd,vnames,maxit),
             "finegray"=fgnet(x,is.sparse,is.tdep,ix,jx,y,ycr,cr,weights,offset,alpha,nobs,
                               nvars,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,isd,vnames,maxit)  )
  if(is.null(lambda))fit$lambda=fix.lam(fit$lambda)##first lambda is infinity; changed to entry point
  fit$call=this.call
  fit$nobs=nobs
  class(fit)=c(class(fit),"phnet")
  fit
}
