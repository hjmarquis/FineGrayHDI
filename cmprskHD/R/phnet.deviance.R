phnet.deviance=function(pred=NULL,y,x=0, ftime,fstatus,offset=pred,weights=NULL,beta=NULL){
  nvars=as.integer(ncol(x))
  is.sparse=FALSE
  ix=jx=NULL
  if(inherits(x,"sparseMatrix")){##Sparse case
    is.sparse=TRUE
    x=as(x,"CsparseMatrix")
    x=as(x,"dgCMatrix")
    ix=as.integer(x@p+1)
    jx=as.integer(x@i+1)
    x=as.double(x@x)
  }else storage.mode(x)="double"
  if(ncol(y)==3 && min(y[,1])==max(y[,1]))
  {
    y[,1]=y[,2]-y[,1]
    y=y[,-2]
    colnames(y)[1]="time"
  }
  is.tdep=(ncol(y)==3)
  is.fg=!is.null(fstatus)
  if(is.fg)
  {
    cr = as.integer(fstatus)
    if(is.tdep)
    {
      if(is.null(ftime))
        stop("Must provide final time (ftime) for Fine-Gray model with
             time dependent covariates. ")
      ycr= as.double(ftime)
    }
  }
  nobs=as.integer(nrow(y))
  if(is.null(weights))weights=as.double(rep(1.0,nobs))
  else weights=as.double(weights)
  if(is.null(offset))offset=as.double(rep(0.0,nobs))
  else offset=as.double(offset)
  if(is.null(beta)){
    beta=double(0)
    nvec=as.integer(1)
    nvars=as.integer(0)
  }
  else{
    beta=as.matrix(beta)
    nvec=as.integer(ncol(beta))
  }

  is.tdep=(ncol(y)==3)
  if(is.tdep)
  {
    if(!is.matrix(y)||!all(match(c("start","stop","status"),dimnames(y)[[2]],0)))
      stop("Proportional Hazards model (with left-truncation or time-dependent covariates)
           requires a matrix with columns 'start', 'stop'
           and 'status'  (binary) as a response; a 'Surv' object suffices",call.=FALSE)
    qy=as.double(y[,"start"])
    ty=as.double(y[,"stop"])
    tevent=as.double(y[,"status"])
    if(any(ty<=qy))stop("exit time before entry time encountered;  not permitted for PH family")
    if(is.fg)
    {
      if(is.sparse)
        fit=.Fortran("spfgLTloglik",
                     nobs,nvars,x,ix,jx,qy,ty,ycr,tevent,cr,offset,
                     weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
      else
        fit=.Fortran("fgLTloglik",
                     nobs,nvars,x,qy,ty,ycr,tevent,cr,offset,
                     weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
    }
    else
    {
      if(is.sparse)
        fit=.Fortran("spcoxLTloglik",
                     nobs,nvars,x,ix,jx,qy,ty,tevent,offset,weights,
                     nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
      else
        fit=.Fortran("coxLTloglik",
                     nobs,nvars,x,qy,ty,tevent,offset,weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
    }
  }
  else
  {
    if(!is.matrix(y)||!all(match(c("time","status"),dimnames(y)[[2]],0)))
      stop("Proportional Hazards model (without left-truncation or time-dependent covariates)
           requires a matrix with columns 'time' (>0)
           and 'status'  (binary) as a response; a 'Surv' object suffices",call.=FALSE)
    ty=as.double(y[,"time"])
    tevent=as.double(y[,"status"])
    if(is.fg)
    {
      if(is.sparse)
        fit=.Fortran("spfgloglik",
                     nobs,nvars,x,ix,jx,ty,tevent,cr,offset,
                     weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
      else
        fit=.Fortran("fgloglik",
                     nobs,nvars,x,ty,tevent,cr,offset,weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
    }
    else
    {
      if(is.sparse)
        fit=.Fortran("spcoxloglik",
                     nobs,nvars,x,ix,jx,ty,tevent,offset,weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
      else
        fit=.Fortran("coxloglik",
                     nobs,nvars,x,ty,tevent,offset,weights,nvec,beta,
                     flog=double(nvec),
                     jerr=integer(1),
                     PACKAGE="cmprskHD")
    }
  }
  if(fit$jerr!=0){
    errmsg=jerr(fit$jerr,maxit=0,pmax=0)
    if(errmsg$fatal)stop(errmsg$msg,call.=FALSE)
    else warning(errmsg$msg,call.=FALSE)
  }

  -2*fit$flog
}



