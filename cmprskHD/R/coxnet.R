coxnet=function(x,is.sparse,is.tdep,ix,jx,y,weights,offset,alpha,nobs,nvars,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,isd,vnames,maxit)
{
  if(is.null(offset)){
    offset=rep(0,nobs)
    is.offset=FALSE}
  else{
    is.offset=TRUE
  }
  storage.mode(offset)="double"
  maxit=as.integer(maxit)
  weights=as.double(weights)

  if(is.tdep)
  {
    if(!is.matrix(y)||!all(match(c("start","stop","status"),dimnames(y)[[2]],0)))
      stop("Cox model (with left-truncation or time-dependent covariates)
            requires a matrix with columns 'start', 'stop'
           and 'status'  (binary) as a response; a 'Surv' object suffices",call.=FALSE)
    qy=as.double(y[,"start"])
    ty=as.double(y[,"stop"])
    tevent=as.double(y[,"status"])
    if(any(ty<=qy))stop("exit time before entry time encountered;  not permitted for Cox family")

    fit=if(is.sparse) .Fortran("spcoxLTnet",
                               parm=alpha,nobs,nvars,x,ix,jx,qy,ty,tevent,offset,weights,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,
                               maxit,isd,# need to get JHF to reverse these
                               lmu=integer(1),
                               ca=double(nx*nlam),
                               ia=integer(nx),
                               nin=integer(nlam),
                               nulldev=double(1),
                               dev=double(nlam),
                               alm=double(nlam),
                               nlp=integer(1),
                               jerr=integer(1),PACKAGE="cmprskHD"
    )
    else .Fortran("coxLTnet",
                  parm=alpha,nobs,nvars,as.double(x),qy,ty,tevent,offset,weights,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,
                  maxit,isd,# need to get JHF to reverse these
                  lmu=integer(1),
                  ca=double(nx*nlam),
                  ia=integer(nx),
                  nin=integer(nlam),
                  nulldev=double(1),
                  dev=double(nlam),
                  alm=double(nlam),
                  nlp=integer(1),
                  jerr=integer(1),PACKAGE="cmprskHD"
    )
  }else
  {
    if(!is.matrix(y)||!all(match(c("time","status"),dimnames(y)[[2]],0)))
      stop("Cox model (without left-truncation or time-dependent covariates)
           requires a matrix with columns 'time' (>0)
           and 'status'  (binary) as a response; a 'Surv' object suffices",call.=FALSE)
    ty=as.double(y[,"time"])
    tevent=as.double(y[,"status"])
    if(any(ty<=0))stop("negative event times encountered;  not permitted for Cox family")

    fit=if(is.sparse) .Fortran("spcoxnet",
                               parm=alpha,nobs,nvars,x,ix,jx,ty,tevent,offset,weights,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,
                               maxit,isd,# need to get JHF to reverse these
                               lmu=integer(1),
                               ca=double(nx*nlam),
                               ia=integer(nx),
                               nin=integer(nlam),
                               nulldev=double(1),
                               dev=double(nlam),
                               alm=double(nlam),
                               nlp=integer(1),
                               jerr=integer(1),PACKAGE="cmprskHD"
    )
      else .Fortran("coxnet",
                    parm=alpha,nobs,nvars,as.double(x),ty,tevent,offset,weights,jd,vp,cl,ne,nx,nlam,flmin,ulam,thresh,
                    maxit,isd,# need to get JHF to reverse these
                    lmu=integer(1),
                    ca=double(nx*nlam),
                    ia=integer(nx),
                    nin=integer(nlam),
                    nulldev=double(1),
                    dev=double(nlam),
                    alm=double(nlam),
                    nlp=integer(1),
                    jerr=integer(1),PACKAGE="cmprskHD"
      )
  }

  if(fit$jerr!=0){
    errmsg=jerr(fit$jerr,maxit,pmax=nx)
    if(errmsg$fatal)stop(errmsg$msg,call.=FALSE)
    else warning(errmsg$msg,call.=FALSE)
  }
  outlist=getcoef(fit,nvars,nx,vnames)
  dev=fit$dev[seq(fit$lmu)]
  outlist=c(outlist,list(dev.ratio=dev,nulldev=fit$nulldev,npasses=fit$nlp,jerr=fit$jerr,offset=is.offset))
  class(outlist)="coxnet"
  outlist
}
