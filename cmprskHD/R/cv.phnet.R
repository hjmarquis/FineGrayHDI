#' Cross-validation for phnet
#'
#' \code{cv.phnet}  Does k-fold cross-validation for phnet, produces a plot,
#' and returns a value for \code{lambda}.
#'
#' @param x \code{x} matrix as in \code{phnet}.
#' @param y response \code{y} as in \code{phnet}.
#' @param ftime final time \code{ftime} as in \code{phnet}.
#' @param fstatus final status \code{fstatus} as in \code{phnet}.
#' Other causes are coded as integers starting from '2'.
#' @param id the subject \code{id} as in \code{phnet}.
#' @param weights observation weights. Default is 1 for each observation
#' @param lambda	Optional user-supplied \code{lambda} sequence;
#' default is \code{NULL}, and \code{phnet} chooses its own sequence.
#' @param nfolds	number of folds - default is 10.
#' Although nfolds can be as large as the sample size (leave-one-out CV),
#' it is not recommended for large datasets. Smallest value allowable is \code{nfolds=3}.
#' @param foldid an optional vector of values between 1 and \code{nfold} identifying what fold
#' each observation is in. If supplied, \code{nfold} can be missing.
#' @param grouped This is an experimental argument, with default \code{TRUE},
#' and can be ignored by most users. \code{grouped=TRUE} obtains
#' the CV partial likelihood for the Kth fold by subtraction;
#' by subtracting the log partial likelihood evaluated on the full dataset
#' from that evaluated on the on the (K-1)/K dataset. This makes more efficient
#' use of risk sets. With \code{grouped=FALSE} the log partial likelihood is
#' computed only on the Kth fold.
#' @param parallel If \code{TRUE}, use parallel \code{foreach} to fit each fold.
#' Must register parallel before hand, such as \code{doMC} or others. See the example below.
#'
#' @details
#' The function runs \code{phnet} \code{nfolds+1} times;
#' the first to get the \code{lambda} sequence,
#' and then the remainder to compute the fit with each of the folds omitted.
#' The error is accumulated, and the average error and standard deviation over
#' the folds is computed.
#' Note that \code{cv.phnet} does NOT search for values for alpha.
#' A specific value should be supplied, else \code{alpha=1} is assumed by default.
#' If users would like to cross-validate \code{alpha} as well,
#' they should call \code{cv.phnet} with a pre-computed vector foldid,
#' and then use this same fold vector in separate calls to \code{cv.phnet}
#'  with different values of \code{alpha}.
#'  Note also that the results of \code{cv.phnet} are random,
#'  since the folds are selected at random. Users can reduce this
#'  randomness by running \code{cv.phnet} many times, and averaging the error curves.
#'
#' @return An object with \code{S3} class \code{"cv.phnet"},
#'  which is a list with the ingredients of the cross-validation fit.
#' \item{lambda}{The values of \code{lambda} used in the fits.}
#' \item{cvm}{The mean cross-validated error - a vector of length \code{length(lambda)}.}
#' \item{cvsd}{The estimate of standard error of \code{cvm}. }
#' \item{cvup}{The upper curve = \code{cvm+cvsd}}
#' \item{cvlo}{The lower curve = \code{cvm-cvsd}}
#' \item{nzero}{The number of non-zero coefficients at each \code{lambda}. }
#' \item{phnet.fit}{The fitted \code{phnet} object for the full data.}
#' \item{lambda.min}{The value of \code{lambda} that gives minimum \code{cvm}.}
#' \item{lambda.1se}{The largest value of \code{lambda} such that error
#' is within 1 standard error of the minimum.}
#' \item{fordid}{The fold assignments used}
#' @author Marquis (Jue) Hou
#' @export cv.phnet
#' @seealso \code{\link[glmnet]{cv.glmnet}}


cv.phnet <-
  function (x, y, ftime=NULL,fstatus=NULL,id,weights, offset = NULL, lambda = NULL,
            nfolds = 10, foldid,
            grouped = TRUE, parallel = FALSE, ...)
  {
    if (!is.null(lambda) && length(lambda) < 2)
      stop("Need more than one value of lambda for cv.phnet")
    N = nrow(x)
    if (missing(weights))
      weights = rep(1, N)
    else weights = as.double(weights)
    if(missing(id)) id=1:N
    y = drop(y)
    phnet.call = match.call(expand.dots = TRUE)
    which.fold = match(c("nfolds", "foldid", "grouped"), names(phnet.call), F)
    if (any(which.fold))
      phnet.call = phnet.call[-which.fold]
    phnet.call[[1]] = as.name("phnet")
    phnet.object = phnet(x, y, ftime=ftime,fstatus=fstatus,id,weights = weights, offset = offset,
                           lambda = lambda, ...)
    phnet.object$call = phnet.call
    is.offset = phnet.object$offset
    ###Next line is commented out so each call generates its own lambda sequence
    # lambda=phnet.object$lambda
    # predict phnet
    nz = sapply(predict(phnet.object, type = "nonzero"),
                     length)
    id.unique = unique(id)
    Nid=length(id.unique)
    if (missing(foldid))
    {
      foldid=rep(0,N)
      foldid.unique= sample(rep(seq(nfolds), length = Nid))
      for(i in 1:Nid)
        foldid[id==id.unique[i]]=foldid.unique[i]
    }
    else nfolds = max(foldid)
    if (nfolds < 3)
      stop("nfolds must be bigger than 3; nfolds=10 recommended")
    outlist = as.list(seq(nfolds))
    if (parallel) {
      #  if (parallel && require(foreach)) {
      outlist = foreach(i = seq(nfolds), .packages = c("cmprskHD")) %dopar%
      {
        i.fold = foldid == i
        id_sub = id[!i.fold]
        #      if (is.matrix(y))
        y_sub = y[!i.fold, ]
        ftime_sub=fstatus_sub=NULL
        if(!is.null(ftime))
          ftime_sub=ftime[!i.fold]
        if(!is.null(fstatus))
          fstatus_sub=fstatus[!i.fold]
        if (is.offset)
          offset_sub = as.matrix(offset)[!i.fold, ]
        else offset_sub = NULL
        phnet(x[!i.fold, , drop = FALSE], y_sub, ftime=ftime_sub,
              fstatus=fstatus_sub,
              id=id_sub,lambda = lambda,
               offset = offset_sub, weights = weights[!i.fold],
               ...)
      }
    }
    else {
      for (i in seq(nfolds)) {
        i.fold = foldid == i
        id_sub = id[!i.fold]
        #      if (is.matrix(y))
        y_sub = y[!i.fold, ]
        x_sub = x[!i.fold, ]
        ftime_sub=fstatus_sub=NULL
        if(!is.null(ftime))
          ftime_sub=ftime[!i.fold]
        if(!is.null(fstatus))
          fstatus_sub=fstatus[!i.fold]
        if (is.offset)
          offset_sub = as.matrix(offset)[!i.fold, ]
        else offset_sub = NULL
        weights_sub = weights[!i.fold]
        outlist[[i]] =phnet(x_sub, y_sub, ftime=ftime_sub,
                            fstatus=fstatus_sub,
                                     id=id_sub,lambda = lambda,
                                     offset = offset_sub, weights = weights_sub,
                                     ...)
      }
    }
    fun = paste("cv", class(phnet.object)[[1]], sep = ".")
    lambda = phnet.object$lambda
    # cv.coxnet, cv.fgnet
    cvstuff = cv.phnet.deviance(outlist, lambda, x, y, ftime,fstatus,weights,
                                offset, foldid, grouped)
    cvm = cvstuff$cvm
    cvsd = cvstuff$cvsd
    nas=is.na(cvsd)
    if(any(nas)){
      lambda=lambda[!nas]
      cvm=cvm[!nas]
      cvsd=cvsd[!nas]
      nz=nz[!nas]
    }
    cvname = cvstuff$name
    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd, cvup = cvm +
                 cvsd, cvlo = cvm - cvsd, nzero = nz, name = cvname, phnet.fit = phnet.object,
               foldid = foldid)
    lamin=if(cvname=="AUC")getmin(lambda,-cvm,cvsd)
    else getmin(lambda, cvm, cvsd)
    obj = c(out, as.list(lamin))
    class(obj) = "cv.phnet"
    obj
  }

