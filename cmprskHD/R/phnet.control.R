#' Internal phnet parameters
#'
#' \code{phnet.control} View and/or change the factory default parameters in \code{phnet}.
#'
#' @param fdev minimum fractional change in deviance for stopping path;
#' factory default = 1.0e-5
#' @param devmax	maximum fraction of explained deviance for stopping path; factory default = 0.999
#' @param eps	minimum value of lambda.min.ratio (see glmnet); factory default= 1.0e-6
#' @param big	large floating point number; factory default = 9.9e35. Inf in definition of upper.limit is set to big
#' @param mnlam	minimum number of path points (lambda values) allowed; factory default = 5
#' @param pmin	minimum probability for any class. factory default = 1.0e-9. Note that this implies a pmax of 1-pmin.
#' @param exmx	maximum allowed exponent. factory default = 250.0
#' @param prec	convergence threshold for multi response bounds adjustment solution. factory default = 1.0e-10
#' @param mxit	maximum iterations for multiresponse bounds adjustment solution. factory default = 100
#' @param factory	If TRUE, reset all the parameters to the factory default; default is FALSE
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
#' @export phnet.control
#' @seealso \code{\link[glmnet]{glmnet.control}}

phnet.control <-
  function (fdev = 1e-05, devmax = 0.999, eps = 1e-06, big = 9.9e+35,
            mnlam = 5, pmin = 1e-09, exmx = 250, prec = 1e-10, mxit = 100,
            factory = FALSE)
  {
    inquiry=!nargs()
    if (factory)
      invisible(phnet.control(fdev = 1e-05, devmax = 0.999,
                               eps = 1e-06, big = 9.9e+35, mnlam = 5, pmin = 1e-09,
                               exmx = 250, prec = 1e-10, mxit = 100))
    else {
      if (!missing(fdev))
        .Fortran("chg_fract_dev", as.double(fdev), PACKAGE = "cmprskHD")
      if (!missing(devmax))
        .Fortran("chg_dev_max", as.double(devmax), PACKAGE = "cmprskHD")
      if (!missing(eps))
        .Fortran("chg_min_flmin", as.double(eps), PACKAGE = "cmprskHD")
      if (!missing(big))
        .Fortran("chg_big", as.double(big), PACKAGE = "cmprskHD")
      if (!missing(mnlam))
        .Fortran("chg_min_lambdas", as.integer(mnlam), PACKAGE = "cmprskHD")
      if (!missing(pmin))
        .Fortran("chg_min_null_prob", as.double(pmin), PACKAGE = "cmprskHD")
      if (!missing(exmx))
        .Fortran("chg_max_exp", as.double(exmx), PACKAGE = "cmprskHD")
      if (!missing(prec) | !missing(mxit))
        .Fortran("chg_bnorm", as.double(prec), as.integer(mxit),
                 PACKAGE = "cmprskHD")
      value=c(.Fortran("get_int_parms", fdev = double(1),
                       eps = double(1), big = double(1), mnlam = integer(1),
                       devmax = double(1), pmin = double(1), exmx = double(1),
                       PACKAGE = "cmprskHD"))
      if(inquiry)value else invisible(value)
    }
  }
