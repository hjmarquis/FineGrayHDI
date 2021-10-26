
ipw.FG <- function(surv, fstatus, id = 1:length(ftime), Z)
{
  n <- nrow(surv)
  if(ncol(surv)==2)
  {
    surv=Surv(rep(0, n),surv[,1], surv[,2])
  }

  if(nrow(Z) != n)
    stop("Z and surv should have the same number of rows. ")

  if(length(id) != n)
    stop("Length of id and surv do not match. ")

  id.unique = unique(id) # assume it is sorted

  id.list = list(1)
  for(i in 2:n)
    if(id[i]==id[i-1])
    {  id.list[[length(id.list)]] = c(id.list[[length(id.list)]],i)
    }else{
      id.list[[length(id.list)+1]] = i
    }

  last.row = sapply(id.list,max)

  if(length(fstatus)==length(id.unique))
  {
    tmp = rep(0, n)
    tmp[last.row] = fstatus
    fstatus = tmp
  }

  if(length(fstatus) != n)
    stop("Length of fstatus does not match the number of rows in surv or the number of unique ids. ")

  if(any(fstatus * c(surv[,3]) != fstatus))
    stop("fstatus should be coded as zero for censored subjects and non-zero for uncensored subjects. ")

  C.Surv=Surv(surv[last.row, 2], 1-surv[last.row, 3])
  C.km <- survfit(C.Surv ~ 1,  type="kaplan-meier")
  C.knot = sort(unique(C.Surv[C.Surv[,2]==1,1]))
  C.surv = c(summary(C.km, times = C.knot, extend = T)$surv)
  if(min(C.surv) > 0)
  {
    C.knot = c(C.knot,max(C.Surv[,1])+1)
    C.surv = c(C.surv, 0)
  }
  Ghat=stepfun(C.knot,c(1,C.surv))

  FG.start = FG.end = FG.event = FG.id = FG.weight = FG.Z = FG.X = NULL



  for(i in 1:length(id.unique))
  {
    ni = length(id.list[[i]])
    id.surv = surv[id.list[[i]],]
    id.Z = id.list[[i]]
    X = id.surv[ni, 2]

    if( ni > 1)
    {
      if(any(c(id.surv[,3]) != c(rep(0, ni-1),1)))
        stop(paste( "Wrong event indication for id = ",
                    id.unique[i],
                    ". Only the last row for each id may have event. "
                    ,sep=''))

      if(any(c(id.surv[,1]) != c(0,id.surv[-ni,2])))
        stop(paste("Wrong coding for start-end time for id = ",
                   id.unique[i],
                   ". The first row of each id must start at 0, ",
                   "and the start of next row must match the end of the previous. ",
                   sep=''))
    }

    if(fstatus[last.row[i]] > 1)
    {
      ck = C.knot[C.knot > X]
      nk=length(ck)
      omega = Ghat(ck)/Ghat(X)
      IPW.weight = -diff(c(1,omega))

      FG.start = c(FG.start, id.surv[,1], rep(id.surv[ni,1], nk-1))
      FG.end = c(FG.end, id.surv[-ni,2],ck)
      FG.event = c(FG.event, rep(0, ni+nk-1))
      FG.weight = c(FG.weight, rep(1,ni-1), IPW.weight)
      #  FG.omega = c(FG.weight, rep(1,ni-1), omega)
      FG.id = c(FG.id, rep(id.unique[i],ni+nk-1))
      FG.Z = c(FG.Z, id.Z[rep(1:ni,each = c(rep(1,ni-1),nk))])
      FG.X = c(FG.X, rep(X,ni+nk-1))
    }else{
      FG.start = c(FG.start, id.surv[,1])
      FG.end = c(FG.end, id.surv[,2])
      FG.event = c(FG.event, id.surv[,3])
      FG.weight = c(FG.weight, rep(1,ni))
      #   FG.omega = c(FG.omega, rep(1,ni))
      FG.id = c(FG.id, rep(id.unique[i],ni))
      FG.Z = c(FG.Z, id.Z)
      FG.X = c(FG.X, rep(X,ni))
    }
  }

  out.surv= Surv(FG.start, FG.end, FG.event)
  out.Z = Z[FG.Z,]

  out <- list(surv = Surv(FG.start, FG.end, FG.event), weight = FG.weight, Z = out.Z,
              X=FG.X, id = FG.id, C.surv = C.Surv)
  class(out) <- "ipw.FG"
  return(out)
}
