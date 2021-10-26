jerr=function(n,maxit,pmax){
  if(n==0) list(n=0,fatal=FALSE,msg="")
  else
  {
    if(n>0)
    {#fatal error
      outlist=jerr.elnet(n)
      if(outlist$msg!="Unknown error")return(outlist)
      if(n==8888)msg="All observations censored - cannot proceed"
      else if(n==9999)msg="No positive observation weights"
      else if(match(n,c(20000,30000),FALSE)) msg="Inititialization numerical error; probably too many censored observations"
      else 	 msg="Unknown error"
      errlist=list(n=n,fatal=TRUE,msg=msg)
    } else
    { # non-fatal error
      if(n<= -30000)
      {
        msg=paste("Numerical error at ",-n-30000,"th lambda value; solutions for larger values of lambda returned",sep="")
        errlist=list(n=n,fatal=FALSE,msg=msg)
      }
      else
      {
        if(n>-10000)msg=paste("Convergence for ",-n,"th lambda value not reached after maxit=",maxit," iterations; solutions for larger lambdas returned",sep="")
        if(n < -10000)msg=paste("Number of nonzero coefficients along the path exceeds pmax=",pmax, " at ",-n-10000,"th lambda value; solutions for larger lambdas returned",sep="")
        errlist=list(n=n,fatal=FALSE,msg=msg)
      }
    }
    names(errlist)=c("n","fatal","msg")
    errlist$msg=paste("from phnet Fortran code (error code ",n, "); ",errlist$msg,sep="")
    errlist
  }
}



