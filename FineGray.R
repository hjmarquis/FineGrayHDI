
#require(survival)
# ---------------------------------------------------------------------------
# Inverse Probability Weighting for Fine-Gray
# ---------------------------------------------------------------------------

# Input: observed data
# Output: IPW survival data
FG.Surv=function(surv,cause, Z, id = 1:nrow(surv))
{
  n=nrow(surv)
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
  
  if(length(cause)==length(id.unique))
  {
    tmp = rep(0, n)
    tmp[last.row] = cause
    cause = tmp
  }
  
  if(length(cause) != n)
    stop("Length of cause does not match the number of rows in surv or the number of unique ids. ")
  
  if(any(cause * c(surv[,3]) != cause))
    stop("Cause should be coded as zero for censored subjects and non-zero for uncensored subjects. ")
  
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
    
    if(cause[last.row[i]] > 1)
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
  
  return(list(surv = Surv(FG.start, FG.end, FG.event), weight = FG.weight, Z = out.Z,
              X=FG.X, id = FG.id, C.surv = C.Surv))
}



FG.score 				= function(Z, Y ,IPW, X, id, C.surv, beta.est){

  id.unique = unique(id)
  n.id = length(id.unique)
  n.row = nrow(Z)
  p = ncol(Z)
  event1 = Y[,3]==1
    
  tk = Y[event1,2] # K
  order.tk = order(tk) # K
  tk = tk[order.tk]
  rr = exp(c(Z%*% beta.est)) # n.row
  
  S.weight = outer(Y[,1],tk, '<') * outer(Y[,2],tk, '>=') * IPW * rr # n.row*K
  S.0 = apply(S.weight, 2, sum) # K
  Z.bar = (t(S.weight)%*% Z)/ S.0 # K*p
  
  dLam1 = 1/S.0 # K
  
  eta.N = matrix(0, n.row, p)
  eta.N[event1,] = Z[event1,] - Z.bar[order.tk,]
  hat_Z_ = apply(eta.N,2,function(x){tapply(x,id,sum)})
  S_ = apply(hat_Z_, 2, mean)
  
  eta.ZLam1 = Z *c(S.weight %*% dLam1) -
          (S.weight %*% (Z.bar*dLam1) )
  
  eta = matrix(0, n.id, p)
  for(i in 1:n.id)
    eta[i, ] = apply(matrix(eta.N[id==id.unique[i],] - eta.ZLam1[id==id.unique[i],],ncol = p), 2, sum)
  
  eventC = C.surv[,2] == 1
  cj = sort(unique(C.surv[eventC,1])) # J
  
  I.censored = outer(C.surv[,1], cj, '==') # n.id*J
  I.atrisk = outer(C.surv[,1], cj, '>=') # n.id*J
  
  dLamc = apply(I.censored,2,sum)/apply(I.atrisk, 2, sum) # J
  pi.j = apply(I.atrisk, 2, mean) # J
  
  q.weight = S.weight * (Y[,2] > X) * outer(X,tk, '<') # n.rwo * K
  dq.k = ((t(q.weight) %*% Z) * dLam1 - Z.bar*dLam1*apply(q.weight,2,sum) )/n.id # K*p
  q.j = -outer(cj, tk, '<=') %*% dq.k # J*p
  
  q.pi.j = q.j/pi.j # J*p
  
  psi.N = matrix(0, n.id, p) # n.id*p
  psi.N[eventC,] = q.pi.j[unlist(apply(I.censored, 1, which)), ]
  
  psi.Lamc = I.atrisk %*% (q.pi.j*dLamc)
  
  psi = psi.N - psi.Lamc
  
  S_iid = eta + psi
  curly_V_ = t(S_iid) %*% S_iid/n.id

  return(list(S_ = S_, hat_Z_ = hat_Z_, curly_V_ = curly_V_))
}

FG.weighted.average 	= function(t, Z, Y, IPW, beta.est){
  
  weight_list_ 	= FG.weight(t = t, Z = Z, Y, IPW=IPW, beta.est = beta.est)
  # computation of s^0 
  sum_weight_ 	= sum(weight_list_)
  if(sum_weight_ == 0){
    return(list(flag = FALSE, ave = NULL))
  }
  # ave saves and returns \bar{Z}
  if(sum_weight_ > 0){
    return(list(flag = TRUE, ave = (1/sum_weight_) * weight_list_ %*% Z))
  }
}


FG.weight 	   	     		 = function(t, Z, Y, IPW, beta.est){
  
  n_ 	   		 		 = nrow(Z)
  weight_list_ 		 = c()
  for(i_ in 1: n_){
    ind_ 	 		 = ifelse(IPW$time2[i_] >= t, 1, 0)
    weight_list_[i_] = ind_ *IPW$weights[i_]* exp(sum(beta.est * Z[i_, ]))
  }
  
  return(weight_list_)
}