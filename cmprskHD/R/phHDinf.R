#' The main function to fit the Fine-Gray model with high-dimensional covariates
#'
#' \code{phHDinf}  returns the inital LASSO estimator, the one-step-bias-correcting estimator
#' and the variance estimator.
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
#' @param inference.Z a matrix whose rows are the linear combinations
#' of the coefficients for inference. Default is the identity matrix.
#' @param init.tuning.method the tuning method for the initial LASSO estimator. Either
#'  "cv", "1se" or "1se.small". "cv" is recommended
#' @param init.par the list of parameters send to "cv.phnet" for the initial
#'  estimator
#' @param inference.tuning.method the tuning method for the one-step-bias-correcting
#'  estimator. Either "cv", "1se" or "1se.small". "cv" is recommended
#' @param inference.par the list of parameters send to "cv.glmnet" for the
#'  one-step-bias-correcting estimator
#' @param var.est the method for the variance estimation. Either "initial", using the
#'  initial estimator, or "onestep", using the one-step-bias-correcting estimator.
#'  "initial" is asymptotically consistent while "onestep" is sometimes empirically better
#'  for smaller sample size.
#' @param ncores number of cores for parallel computing
#' @details
#' \code{fsurv} can have either 2 components for fixed covariates, sojourn time and status, or 3 components
#' for time-dependent covariates, entry time, exit time and status. The status in \code{fsurv} is coded
#' as "1" for the final entry of any type of observed event while "0" stands either for a non-final entry
#' or the final entry of an observed censoring. Similarly, the non-final entries should be coded as
#' the value of \code{cencode} in \code{fstatus}.
#' For the efficiency of the algorithm, we require that
#' the subjects are acsendingly sorted by their id, and
#' the entries for the same subject are consecutive and ordered in time from early to late.
#'
#' @return a phHDinf object
#' \item{initial}{The initial LASSO estimator}
#' \item{onestep}{The one-step-bias-correcting estimator}
#' \item{var}{The variance estimator for the one-step-bias-correcting estimator}
#' @author Marquis (Jue) Hou
#' @export phHDinf
#' @seealso \code{\link[phnet]{cv.phnet}}
#' @seealso \code{\link[glmnet]{cv.glmnet}}





phHDinf <- function(x,y,ftime=NULL,fstatus=NULL,id=1:nrow(x),family=c("cox","finegray"),
                    weights,offset=NULL,
                     inference.Z,
                     init.tuning.method = c('cv','1se'),
                     init.par = list(),
                     inference.tuning.method = c('cv','1se'),
                     inference.par = list(),
                     var.est = c("initial", "onestep"),
                     ncores = 1)
{
  if(ncores>1)
  {
    cl=makeCluster(getOption("cl.cores", ncores))
    registerDoParallel(cl)
    parallel=TRUE
  }
  nvars=ncol(x)
  if(missing(inference.Z))
  {
    if(inherits(x,"sparseMatrix"))
      inference.Z=Diagonal(nvars,1)
    else
      inference.Z=diag(1,nvars)
  }
  if(ncol(inference.Z)!=nvars)
    stop("inference.Z must be a matrix with the same number of columns as x.")
  if(var.est!="initial"&&var.est!="onestep")
    stop("var.est must be one of \"initial\" or \"onestep\"")
  if(init.tuning.method!="cv"&&init.tuning.method!="1se")
    stop("Tuning method for initial estimator can only be either \"cv\" or \"1se\".")
  if(inference.tuning.method!="cv"&&inference.tuning.method!="1se")
    stop("Tuning method for onestep estimator can only be either \"cv\" or \"1se\".")
  nobs=nrow(x)
  if(length(id)!=x)
    stop("The length of id should equal the number of entries in the data.")
  id.order=order(id)
  x=x[id.order,]
  y=y[id.order]
  id=id[id.order]

  init.fit = do.call(cv.phnet,c(list(x=x,y=y,
              ftime=ftime,fstatus=fstatus,id=id,weights=weights,
              offset=offset),init.glmnet.par))
  if(init.tuning.method=="cv")
    initial=coef(init.fit$phnet.fit,s=init.fit$lambda.min)
  else
    initial=coef(init.fit$phnet.fit,s=init.fit$lambda.1se)

  out=list()
  out$initial = drop(inference.Z %*% initial)

  inference=do.call(ph.nodewise,c(list(x=x,y=y,
               ftime=ftime,fstatus=fstatus,id=id,weights=weights,
               offset=offset, beta=inital),init.glmnet.par))
  out$onestep = drop(inference.Z %*% inference$onestep)
  if(var.est=="onestep")
    inference=do.call(ph.nodewise,c(list(x=x,y=y,
              ftime=ftime,fstatus=fstatus,id=id,weights=weights,
              offset=offset, beta=inference$onestep),init.glmnet.par))

  out$var=inference.Z%*%inference$var%*%t(inference.Z)
  class(out)<-"phHDinf"
  out
}
