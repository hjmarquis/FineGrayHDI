#' Make predictions from a "phnet" object.
#'
#' Similar to other predict methods,
#' this functions predicts fitted values, logits, coefficients
#' and more from a fitted \code{"phnet"} object.
#'
#'
#' @param object Fitted \code{phnet} matrix as in \code{phnet}.
#' @param newx Matrix of new values for x at which predictions are to be made.
#' Must be a matrix; can be sparse as in \code{Matrix} package.
#' This argument is not used for \code{type=c("coefficients","nonzero")}.
#' @param s Value(s) of the penalty parameter \code{lambda} at which predictions are required.
#' Default is the entire sequence used to create the model.
#' @param type Type of prediction required. Type \code{"link"} gives the linear predictors.
#' Type \code{"response"} gives the fitted relative-risk.
#' Type \code{"coefficients"} computes the coefficients at the requested values for \code{s}.
#' Type \code{"nonzero"} returns a list of the indices of the nonzero coefficients
#' for each value of \code{s}.
#' @param exact This argument is relevant only when predictions are made at values of
#' \code{s} (lambda) different from those used in the fitting of the original model.
#' If \code{exact=FALSE} (default), then the predict function uses linear interpolation
#' to make predictions for values of \code{s} (lambda) that do not coincide
#' with those used in the fitting algorithm.
#' While this is often a good approximation, it can sometimes be a bit coarse.
#' With \code{exact=TRUE}, these different values of \code{s} are merged (and sorted) with
#' \code{object$lambda}, and the model is refit before predictions are made.
#' In this case, it is required to supply the original data \code{x=} and \code{y=} as
#' additional named arguments to \code{predict()} or \code{coef()}.
#' The workhorse \code{predict.glmnet()} needs to update the model,
#' and so needs the data used to create it. The same is true of
#' \code{weights, offset, penalty.factor, lower.limits, upper.limits}
#'  if these were used in the original call. Failure to do so will result in an error.
#' @param newoffset If an offset is used in the fit, then one must be supplied
#' for making predictions (except for \code{type="coefficients"} or \code{type="nonzero"}).
#' @param ...	This is the mechanism for passing arguments
#' like \code{x=} when \code{exact=TRUE}; see \code{exact} argument.
#'
#'
#' @return The object returned depends on type.
#' @author Marquis (Jue) Hou
#' @export predict.phnet
#' @seealso \code{\link[glmnet]{predict.glmnet}}


predict.phnet=function(object,newx,s=NULL,type=c("link","response","coefficients","nonzero"),exact=FALSE,offset,...){
  type=match.arg(type)
  ###coxnet has no intercept, so we treat it separately
  if(missing(newx)){
    if(!match(type,c("coefficients","nonzero"),FALSE))stop("You need to supply a value for 'newx'")
  }
  if(exact&&(!is.null(s))){
    lambda=object$lambda
    which=match(s,lambda,FALSE)
    if(!all(which>0)){
      lambda=unique(rev(sort(c(s,lambda))))
      object=tryCatch(update(object,lambda=lambda,...),error=function(e)stop("problem with predict.phnet() or coef.phnet(): unable to refit the phnet object to compute exact coefficients; please supply original data by name, such as x and y, plus any weights, offsets etc.",call.=FALSE))
    }
  }

  nbeta=object$beta
  if(!is.null(s)){
    vnames=dimnames(nbeta)[[1]]
    dimnames(nbeta)=list(NULL,NULL)
    lambda=object$lambda
    lamlist=lambda.interp(lambda,s)
    nbeta=nbeta[,lamlist$left,drop=FALSE]%*%Diagonal(x=lamlist$frac) +nbeta[,lamlist$right,drop=FALSE]%*%Diagonal(x=1-lamlist$frac)
    dimnames(nbeta)=list(vnames,paste(seq(along=s)))
  }
  if(type=="coefficients")return(nbeta)
  if(type=="nonzero")return(nonzeroCoef(nbeta,bystep=TRUE))
  nfit=as.matrix(newx%*%nbeta)
  if(object$offset){
    if(missing(offset))stop("No offset provided for prediction, yet used in fit of phnet",call.=FALSE)
    nfit=nfit+array(offset,dim=dim(nfit))
  }
  switch(type,
         response=exp(nfit),
         link=nfit
  )
}

#' @rdname predict.phnet
#' @export coef.phnet
coef.phnet=function(object,s=NULL,exact=FALSE,...)
  predict(object,s=s,type="coefficients",exact=exact,...)
