#' @export test.groups
test.groups=function()
{
  no=as.integer(10)
  yt=as.double(rep(1:5,each=2))
  d=as.double(c(0,0,0,1,0,1,0,0,0,1))
  q=as.double(rep(1,no))
  nk=as.integer(0)
  kp=as.integer(1:10)
  jp=as.integer(1:10)
  t0=as.double(0)
  jerr=as.integer(0)

  return(.Fortran("groups",no,yt,d,q,nk,kp,jp,t0,jerr))
}

#' @export test.usk
test.usk = function()
{
  no=as.integer(10)
  nk=as.integer(3)
  kp=as.integer(c(2,6,8,4,5,6,7,8,9,10))
  jp=as.integer(c(4,3,6,5,7,8,9,10,9,10))
  u=as.double(rep(0,nk))
  e=as.double(1:10)

  return(.Fortran("usk",no,nk,kp,jp,e,u))
}

#' @export test.risk
test.risk=function()
{
  no=as.integer(10)
  ni=as.integer(100)
  nk=as.integer(3)
  d=as.double(c(0,0,0,1,0,1,0,0,0,1))
  dk=as.double(c(1,1,1))
  f=as.double(log(1:10))
  e=as.double(1:10)
  kp=as.integer(c(2,6,8,4,5,6,7,8,9,10))
  jp=as.integer(c(4,3,6,5,7,8,9,10,9,10))
  u=as.double(rep(0,nk))

  return(.Fortran("risk",no,ni,nk,d,dk,f,e,kp,jp,u))
}

#' @export test.outer
test.outer=function()
{
  no=as.integer(10)
  nk=as.integer(3)
  d=as.double(c(0,0,0,1,0,1,0,0,0,1))
  dk=as.double(c(1,1,1))
  kp=as.integer(c(2,6,8,4,5,6,7,8,9,10))
  jp=as.integer(c(4,3,6,5,7,8,9,10,9,10))
  e=as.double(1:10)
  wr=as.double(rep(0,10))
  w=as.double(rep(0,10))
  jerr=as.integer(0)
  u=as.double(rep(0,nk))

  return(.Fortran("outer",no,nk,d,dk,kp,jp,e,
                  wr,w,jerr,u))
}

#' @export test.groupsLT
test.groupsLT=function(y)
{
  no=as.integer(nrow(y))
  yq=as.double(y[,"start"])
  yt=as.double(y[,"stop"])
  d=as.double(y[,"status"])
  q=as.double(rep(1,no))
  nk=as.integer(0)
  kp=as.integer(1:no)
  jp=as.integer(1:no)
  kq=as.integer(1:no)
  jq=as.integer(1:no)
  t0=as.double(0)
  tk=as.double(0)
  jerr=as.integer(0)

  return(.Fortran("groupsLT",no,yq,yt,d,q,nk,kp,jp,
                  kq,jq,t0,tk,jerr))
}

#' @export test.uskLT
test.uskLT = function(y)
{
  out.group = test.groupsLT(y)
  no=as.integer(out.group[[1]])
  nk=as.integer(out.group[[6]])
  kp=as.integer(out.group[[7]])
  jp=as.integer(out.group[[8]])
  kq=as.integer(out.group[[9]])
  jq=as.integer(out.group[[10]])
  u=as.double(rep(0,nk))
  e=as.double(rep(1,no))

  return(.Fortran("uskLT",no,nk,kp,jp,kq,jq,e,u))
}

#' @export test.outerLT
test.outerLT=function(y)
{
  out.group = test.groupsLT(y)
  out.u = test.uskLT(y)
  no=as.integer(out.group[[1]])
  nk=as.integer(out.group[[6]])
  d=as.double(out.group[[4]])
  kp=as.integer(out.group[[7]])
  jp=as.integer(out.group[[8]])
  dk=as.double(rep(0,no))
  out.d = .Fortran("died",no,nk,d,kp,jp,dk)
  dk=as.double(out.d[[6]])
  kq=as.integer(out.group[[9]])
  jq=as.integer(out.group[[10]])
  u=as.double(out.u[[8]])
  e=as.double(rep(1,no))
  wr=as.double(rep(0,no))
  w=as.double(rep(0,no))
  jerr=as.integer(0)

  return(.Fortran("outerLT",no,nk,d,dk,kp,jp,kq,jq,
                  e,wr,w,jerr,u))
}

#' @export test.groupsFG
test.groupsFG=function(y=Surv(rep(1:5,each=2),
                              c(1,0,0,1,0,0,0,0,0,1)),
                       cr=c(1,0,0,1,0,2,0,0,0,1))
{
  no=as.integer(nrow(y))
  yt=as.double(y[,"time"])
  cr=as.integer(cr)
  q=as.double(rep(1,no))
  nk=as.integer(0)
  kp=as.integer(1:no)
  jp=as.integer(1:no)
  kcr=as.integer(1:no)
  jcr=as.integer(1:no)
  km=as.double(rep(0,no))
  t0=as.double(0)
  jerr=as.integer(0)

  return(.Fortran("groupsFG",no,yt,cr,q,nk,kp,jp,
                  kcr,jcr,km,t0,jerr))
}

#' @export test.uskFG
test.uskFG=function()
{
  tmp=test.groupsFG()
  no=as.integer(tmp[[1]])
  nk=as.integer(tmp[[5]])
  kp=as.integer(tmp[[6]])
  jp=as.integer(tmp[[7]])
  kcr=as.integer(tmp[[8]])
  jcr=as.integer(tmp[[9]])
  km=as.double(tmp[[10]])
  e=as.double(rep(1,no))
  u=as.double(rep(0,no))

  return(.Fortran("uskFG",no,nk,kp,jp,kcr,jcr,km,e,u))
}

#' @export test.outerFG
test.outerFG=function(y=Surv(rep(1:5,each=2),
                             c(0,0,0,1,0,1,0,0,0,1)),
                      cr=c(0,0,0,1,0,1,0,0,0,1),
                      e=rep(1,nrow(y)))
#                             c(1,0,0,1,0,0,0,0,0,1)),
#                      cr=c(1,0,0,1,0,2,0,0,0,1))
{
  tmp=test.groupsFG(y,cr)
  no=as.integer(tmp[[1]])
  nk=as.integer(tmp[[5]])
  d = as.double(tmp[[3]]==1)
  dq = d
  dk=as.double(1:no)
  kp=as.integer(tmp[[6]])
  jp=as.integer(tmp[[7]])
  kcr=as.integer(tmp[[8]])
  jcr=as.integer(tmp[[9]])
  km=as.double(tmp[[10]])
  e=as.double(e)
  wr=as.double(rep(0,no))
  w=as.double(rep(0,no))
  jerr=as.integer(0)
  u=as.double(rep(0,no))
  tmp2=.Fortran("died",no,nk,dq,kp,jp,dk)
  dk=as.double(tmp2[[6]])

  return(.Fortran("outerFG",no,nk,dq,dk,kp,jp,kcr,jcr,km,e,wr,
                  w,jerr,u))
}

#' @export test.groupsFGLT
test.groupsFGLT=function(y,cr,ft)
{
  no=as.integer(nrow(y))
  yq=as.double(y[,"start"])
  yt=as.double(y[,"stop"])
  cr=as.integer(cr)
  ft=as.double(ft)
  q=as.double(rep(1,no))
  nk=as.integer(0)
  kp=as.integer(1:no)
  jp=as.integer(1:no)
  kq=as.integer(1:no)
  jq=as.integer(1:no)
  kcr=as.integer(1:no)
  jcr=as.integer(1:no)
  kpcr=as.integer(1:no)
  jpcr=as.integer(1:no)
  kqcr=as.integer(1:no)
  jqcr=as.integer(1:no)
  km=as.double(rep(0,no))
  t0=as.double(0)
  jerr=as.integer(0)

  return(.Fortran("groupsFGLT",no,yq,yt,ft,cr,q,nk,kp,jp,kq,jq,
                  kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,t0,jerr))
}

#' @export test.uskFGLT
test.uskFGLT=function(y,cr,ft)
{
  tmp=test.groupsFGLT(y,cr,ft)
  no=as.integer(tmp[[1]])
  nk=as.integer(tmp[[7]])
  kp=as.integer(tmp[[8]])
  jp=as.integer(tmp[[9]])
  kq=as.integer(tmp[[10]])
  jq=as.integer(tmp[[11]])
  kcr=as.integer(tmp[[12]])
  jcr=as.integer(tmp[[13]])
  kpcr=as.integer(tmp[[14]])
  jpcr=as.integer(tmp[[15]])
  kqcr=as.integer(tmp[[16]])
  jqcr=as.integer(tmp[[17]])
  km=as.double(tmp[[18]])
  e=as.double(rep(1,no))
  u=as.double(rep(0,no))

  return(.Fortran("uskFGLT",no,nk,kp,jp,kq,jq,kcr,jcr,
                  kpcr,jpcr,kqcr,jqcr,km,e,u))
}

#' @export test.outerFGLT
test.outerFGLT=function(y,cr,ft,e=rep(1,nrow(y)))
{
  tmp=test.groupsFGLT(y,cr,ft)
  no=as.integer(nrow(y))
  yq=as.double(y[,"start"])
  yt=as.double(y[,"stop"])
  cr=as.integer(cr)
  ft=as.double(ft)
  nk=as.integer(tmp[[7]])
  d = as.double(y[,"status"])
  dq = d
  dk=as.double(1:no)
  kp=as.integer(tmp[[8]])
  jp=as.integer(tmp[[9]])
  kq=as.integer(tmp[[10]])
  jq=as.integer(tmp[[11]])
  kcr=as.integer(tmp[[12]])
  jcr=as.integer(tmp[[13]])
  kpcr=as.integer(tmp[[14]])
  jpcr=as.integer(tmp[[15]])
  kqcr=as.integer(tmp[[16]])
  jqcr=as.integer(tmp[[17]])
  km=as.double(tmp[[18]])
  e=as.double(e)
  wr=as.double(rep(0,no))
  w=as.double(rep(0,no))
  jerr=as.integer(0)
  u=as.double(test.uskFGLT(y,cr,ft)[[15]])
  tmp2=.Fortran("died",no,nk,dq,kp,jp,dk)
  dk=as.double(tmp2[[6]])

  return(.Fortran("outerFGLT",no,nk,dq,dk,kp,jp,kq,jq,
                  kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,e,wr,
                  w,jerr,u))
}
