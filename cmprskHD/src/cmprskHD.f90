!          Elastic net with Cox proportional hazards model
!
! dense predictor matrix (no left-truncation):
!
! call coxnet (parm,no,ni,x,yt,d,o,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
!              maxit,isd,lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
!
! dense predictor matrix (allow left-truncation):
!
! call coxLTnet (parm,no,ni,x,yq,yt,d,o,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,
!              maxit,isd,lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
!
!
! sparse predictor matrix (no left-truncation):
!
! call spcoxnet (parm,no,ni,x,ix,jx,yt,d,o,w,jd,vp,cl,ne,nx,nlam,flmin
!              ,ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
!
! sparse predictor matrix (allow left-truncation):
!
! call spcoxLTnet (parm,no,ni,x,ix,jx,yq,yt,d,o,w,jd,vp,cl,ne,nx,nlam,flmin,
!              ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr)
!
! input:
!
!   x(no,ni) = predictor data matrix flat file (overwritten)
!   yt(no) = exit times
!   d(no) = died/censored indicator
!       d(i)=0.0d0 => yt(i) = censoring time
!       d(i)=1.0d0 => yt(i) = death time
!   o(no) = observation off-sets
!   parm = penalty member index (0 <= parm <= 1)
!        = 0.0d0 => ridge
!        = 1.0d0 => lasso
!   no = number of observations
!   ni = number of predictor variables
!   w(no)= observation weights (overwritten)
!   jd(jd(1)+1) = predictor variable deletion flag
!      jd(1) = 0  => use all variables
!      jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
!   vp(ni) = relative penalties for each predictor variable
!      vp(j) = 0 => jth variable unpenalized
!   cl(2,ni) = interval constraints on coefficient values (overwritten)
!      cl(1,j) = lower bound for jth coefficient value (<= 0.0d0)
!      cl(2,j) = upper bound for jth coefficient value (>= 0.0d0)
!   ne = maximum number of variables allowed to enter largest model
!        (stopping criterion)
!   nx = maximum number of variables allowed to enter all models
!        along path (memory allocation, nx > ne).
!   nlam = (maximum) number of lamda values
!   flmin = user control of lamda values (>=0)
!      flmin < 1.0d0 => minimum lamda = flmin*(largest lamda value)
!      flmin >= 1.0d0 => use supplied lamda values (see below)
!   ulam(nlam) = user supplied lamda values (ignored if flmin < 1.0d0)
!   thr = convergence threshold for each lamda solution.
!      iterations stop when the maximum reduction in the criterion value
!      as a result of each parameter update over a single pass
!      is less than thr times the null criterion value.
!      (suggested value, thr=1.0e-5)
!   maxit = maximum allowed number of passes over the data for all lambda
!      values (suggested values, maxit = 100000)
!
! output:
!
!   lmu = actual number of lamda values (solutions)
!   ca(nx,lmu) = compressed coefficient values for each solution
!   ia(nx) = pointers to compressed coefficients
!   nin(lmu) = number of compressed coefficients for each solution
!   dev0 = null deviance (intercept only model)
!   fdev(lmu) = fraction of devience explained by each solution
!   nlp = total number of passes over predictor variables
!   jerr = error flag
!      jerr = 0  => no error - output returned
!      jerr > 0 => fatal error - no output returned
!         jerr < 7777 => memory allocation error
!         jerr = 7777 => all used predictors have zero variance
!         jerr = 8888 => all observations censored (d(i)=0.0d0)
!         jerr = 9999 => no positive observations weights
!         jerr = 10000 => maxval(vp) <= 0.0d0
!         jerr = 20000, 30000 => initialization numerical error
!      jerr < 0 => non fatal error - partial output:
!         Solutions for larger lamdas (1:(k-1)) returned.
!         jerr = -k => convergence for kth lamda value not reached
!            after maxit (see above) iterations.
!         jerr = -10000-k => number of non zero coefficients along path
!            exceeds nx (see above) at kth lamda value.
!         jerr = -30000-k => numerical error at kth lambda value
!    o(no) = training data values for last (lmu_th) solution linear
!            combination.
!
!
!

!
!   yq(no) = entry times
!
!   parm,no,ni,x,yq,yt,d,o,w,jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,maxit,isd,
!   lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr = same as coxnet
!
! output:
!   lmu,ca,ia,nin,dev0,fdev,alm,nlp,jerr,o = same as coxnet above
!
! coxLTnet utility routines:
!
!
! uncompress coefficient vectors for all solutions:
!
! call solns(ni,nx,lmu,ca,ia,nin,b)
!
! input:
!
!    ni,nx = input to coxnet
!    lmu,ca,ia,nin = output from coxnet
!
! output:
!
!    b(ni,lmu) = all coxnet returned solutions in uncompressed format
!
!
! uncompress coefficient vector for particular solution:
!
! call uncomp(ni,ca,ia,nin,a)
!
! input:
!
!    ni = total number of predictor variables
!    ca(nx) = compressed coefficient values for the solution
!    ia(nx) = pointers to compressed coefficients
!    nin = number of compressed coefficients for the solution
!
! output:
!
!    a(ni) =  uncompressed coefficient vector
!             referencing original variables
!
!
!
! evaluate linear model from compressed coefficients and
! uncompressed predictor matrix:
!
! call cxmodval(ca,ia,nin,n,x,f);
!
! input:
!
!    ca(nx) = compressed coefficient values for a solution
!    ia(nx) = pointers to compressed coefficients
!    nin = number of compressed coefficients for solution
!    n = number of predictor vectors (observations)
!    x(n,ni) = full (uncompressed) predictor matrix
!
! output:
!
!    f(n) = model predictions
!
!
! compute log-likelihood for given data set and vectors of coefficients
!
! call loglike(no,ni,x,yq,yt,d,o,w,nvec,a,flog,jerr)
!
! input:
!
!   no = number of observations
!   ni = number of predictor variables
!   x(no,ni) = predictor data matrix flat file
!   yq(no) = entry times
!   yt(no) = exit times
!   d(no) = died/censored indicator
!       d(i)=0.0d0 => y(i) = censoring time
!       d(i)=1.0d0 => y(i) = death time
!   o(no) = observation off-sets
!   w(no)= observation weights
!   nve! = number of coefficient vectors
!   a(ni,nvec) = coefficient vectors (uncompressed)
!
! output
!
!   flog(nvec) = respective log-likelihood values
!   jerr = error flag - see coxnet above
!


! coxnet and auxiliaries: Label 30000~39999
! main double precision function: initialization and return
! Label 30000~30999
      subroutine coxnet(parm,no,ni,x,yt,d,g,w,jd,vp,cl,ne,nx,nlam,&
flmin,ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer jd(*),ia(nx),nin(nlam)
      double precision, dimension (:), allocatable :: xs,ww,vq
      integer, dimension (:), allocatable :: ju

      if(maxval(vp) .gt. 0.0d0)goto 30001
      jerr=10000
      return
30001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 30101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
30101 continue
      if(jerr.ne.0) return
      call chkvars(no,ni,x,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 30201
      jerr=7777
      return
30201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 30301
      jerr=9999
      return
30301 continue
      ww=ww/sw
      call cstandard(no,ni,x,ww,ju,isd,xs)
      if(isd .le. 0)goto 30499
30400 do 30401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
30401 continue
30450 continue
30499 continue
      call coxnet1(parm,no,ni,x,yt,d,g,ww,ju,vq,cl,ne,nx,nlam,&
flmin,ulam,thr,maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 30999
30600 do 30601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
30601 continue
30699 continue
30999 continue
      deallocate(ww,ju,vq)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 31000~33999
      subroutine coxnet1(parm,no,ni,x,yt,d,g,q,ju,vp,cl,ne,nx,nlam,&
        flmin,ulam,cthri,maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),q(no),g(no),vp(ni),ulam(nlam)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ju(ni),m(nx),kin(nlam)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,mm,ixx
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnet code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 31999
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) go to 31999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 31011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 31021
31011 continue
      f=0.0d0
      e=q
31021 continue
31101 continue
      r0=risk(no,nk,dq,dk,f,e,kp,jp,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
31200 do 31201 i=1,no
!      if((yt(i) .ge. t0) .and. (q(i) .gt. 0.0d0))goto 31250
      w(i)=0.0d0
      wr(i)=w(i)
31250 continue
31201 continue
31299 continue
      call outer(no,nk,dq,dk,kp,jp,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 31999
      if(flmin .ge. 1.0d0)goto 31301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
31301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
31400 do 31401 j=1,ni
      if(ju(j).eq.0)goto 31401
      ga(j)=abs(dot_product(wr,x(:,j)))
31401 continue
31499 continue
32000 do 32001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 32011
      al=ulam(ilm)
      goto 32099
32011 if(ilm .le. 2)goto 32021
      al=al*alf
      goto 32099
32021 if(ilm .ne. 1)goto 32031
      al=big
      goto 32098
32031 continue
      al0=0.0d0
32100 do 32101 j=1,ni
      if(ju(j).eq.0)goto 32101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
32101 continue
32199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
32098 continue
32099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
32200 do 32201 k=1,ni
      if(ixx(k).eq.1)goto 32201
      if(ju(k).eq.0)goto 32201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
32201 continue
32299 continue
33000 continue
33001 continue
33002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call vars(no,ni,x,w,ixx,v)

! Loop starts
33100 continue
33101 continue
      nlp=nlp+1
      dli=0.0d0
33200 do 33201 j=1,ni
      if(ixx(j).eq.0)goto 33201
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 33211
      at=0.0d0
      goto 33221
33211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/ &
      (v(j)+vp(j)*omal)))
33221 continue
33231 continue
      if(at .eq. a(j))goto 33249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
      if(mm(j) .ne. 0)goto 33248
      nin=nin+1
      if(nin.gt.nx)goto 33299
      mm(j)=nin
      m(nin)=j
33248 continue
33249 continue
33201 continue
33299 continue
      if(nin.gt.nx)goto 33400
      if(dli.lt.cthr)goto 33400
      if(nlp .le. maxit)goto 33300
      jerr=-ilm
      return

33300 continue
33301 continue
33302 continue
      nlp=nlp+1
      dli=0.0d0
33310 do 33311 l=1,nin
      j=m(l)
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 33312
      at=0.0d0
      goto 33313
33312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
      (v(j)+vp(j)*omal)))
33313 continue
33314 continue
      if(at .eq. a(j))goto 33315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
33315 continue
33311 continue
33319 continue
      if(dli.lt.cthr)goto 33331
      if(nlp .le. maxit)goto 33321
      jerr=-ilm
      return
33321 continue
      goto 33302
33331 continue
      goto 33101
33400 continue
      if(nin.gt.nx)goto 33499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outer(no,nk,dq,dk,kp,jp,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 33411
      jerr=jerr-ilm
      go to 31999
33411 continue
      ix=0
33420 do 33421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 33421
      ix=1
      goto 33429
33421 continue
33429 continue
      if(ix .ne. 0)goto 33498
33430 do 33431 k=1,ni
      if(ixx(k).eq.1)goto 33431
      if(ju(k).eq.0)goto 33431
      ga(k)=abs(dot_product(wr,x(:,k)))
      if(ga(k) .le. sa*vp(k))goto 33435
      ixx(k)=1
      ix=1
33435 continue
33431 continue
33439 continue
      if(ix.eq.1) go to 33000
      goto 33499
33498 continue
      goto 33002
33499 continue
      if(nin .le. nx)goto 33500
      jerr=-10000-ilm
      goto 32999
33500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(risk(no,nk,dq,dk,f,e,kp,jp,uu)-r0)/rr
      if(ilm.lt.mnl)goto 32001
      if(flmin.ge.1.0d0)goto 32001
      me=0
33510 do 33511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
33511 continue
33999 continue
      if(me.gt.ne)goto 32999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 32999
      if(dev(ilm).gt.devmax)goto 32999
32001 continue
32999 continue
      g=f
31999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx)
      return
      end

! at-risk information
! Label: 34000~34999
     subroutine groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      implicit double precision (a-h,o-z)
      double precision yt(no),q(no),d(no)
      integer jp(no),kp(*)
34010 do 34011 j=1,no
      jp(j)=j
34011 continue
34021 continue
! Get the order of yt, jp
      call psort7(yt,jp,1,no)
      nj=0
! Remove entries with non-positive weights from the order jp
! nj is the number of positive weights
34100 do 34101 j=1,no
      if(q(jp(j)).le.0.0d0)goto 34101
      nj=nj+1
      jp(nj)=jp(j)
34101 continue
34198 continue
      if(nj .ne. 0)goto 34199
! Error if no positive weights
      jerr=20000
      return
34199 continue
! j index for first event time in jp
      j=1
34200 continue
34201 if(d(jp(j)).gt.0.0d0)goto 34209
      j=j+1
      if(j.gt.nj)goto 34209
      goto 34201
34209 continue
      if(j .lt. nj-1)goto 34299
! Error if no event has positive weight
      jerr=30000
      return
34299 continue
! t0 first event time
! j0 index before t0
      t0=yt(jp(j))
      j0=j-1

! Remove entries censored prior to the first event time
! Need to consider ties at t0
      if(j0 .le. 0)goto 34499
34400 continue
34401 if(yt(jp(j0)).lt.t0)goto 34409
      j0=j0-1
      if(j0.eq.0)goto 34409
      goto 34401
34409 continue
      if(j0 .le. 0)goto 34498
      nj=nj-j0
34420 do 34421 j=1,nj
      jp(j)=jp(j+j0)
34421 continue
34429 continue
34498 continue
34499 continue

! nk number of unique event times
! kp mark the last index for each groups of unique event times in jp
! j index in jp
! jj index in jq
      jerr=0
      nk=0
      yk=t0
      j=2

34500 continue
34600 continue
34601 if(d(jp(j)).gt. 0.0d0 .and. yt(jp(j)).gt.yk)goto 34609
      j=j+1
      if(j.gt.nj)goto 34609
      goto 34601
34609 continue
      nk=nk+1
      kp(nk)=j-1
      if(j.gt.nj)goto 34999
34611 continue
      yk=yt(jp(j))
! Search for potential ties with yk sorted to the front
      jjj=j-1
      if(yt(jp(jjj)) .lt. yk)goto 34699
34620 jjj=jjj-1
      if(yt(jp(jjj)) .ge. yk)goto 34620
      kp(nk)=jjj
34699 continue
      j=j+1
      if(j.le.nj)goto 34500
      nk=nk+1
      kp(nk)=nj
34999 continue
      return
      end

! log-likelihood
! Label: no
      double precision function risk(no,nk,d,dk,f,e,kp,jp,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),f(no)
      integer kp(nk),jp(no)
      double precision e(no),u(nk)
      call usk(no,nk,kp,jp,e,u)
      risk=dot_product(d,f)-dot_product(dk,log(u))
      return
      end

! total risks at event times
! Label: 35000~35999
      subroutine usk(no,nk,kp,jp,e,u)
      implicit double precision (a-h,o-z)
      double precision e(no),u(nk),h
      integer kp(nk),jp(no)
      h=0.0d0
35100 do 35190 k=nk,1,-1
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
35120 do 35121 j=j2,j1,-1
      h=h+e(jp(j))
35121 continue
      u(k)=h
35190 continue
35199 continue
      return
      end

! two quantities for coordinate descent
! Label: 35200~35999
      subroutine outer(no,nk,d,dk,kp,jp,e,wr,w,jerr,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),wr(no),w(no)
      double precision e(no),u(no),b,c
      integer kp(nk),jp(no)
      call usk(no,nk,kp,jp,e,u)
      b=dk(1)/u(1)
      c=dk(1)/u(1)**2
      jerr=0
35200 do 35201 j=1,kp(1)
      i=jp(j)
      w(i)=e(i)*(b-e(i)*c)
      if(w(i) .gt. 0.0d0)goto 35211
      jerr=-30000
      return
35211 continue
      wr(i)=d(i)-e(i)*b
35201 continue
35299 continue
35300 do 35301 k=2,nk
      j1=kp(k-1)+1
      j2=kp(k)
      b=b+dk(k)/u(k)
      c=c+dk(k)/u(k)**2
35340 do 35341 j=j1,j2
      i=jp(j)
      w(i)=e(i)*(b-e(i)*c)
      if(w(i) .gt. 0.0d0)goto 35345
      jerr=-30000
      return
35345 continue
      wr(i)=d(i)-e(i)*b
35341 continue
35349 continue
35301 continue
35999 continue
      return
      end

! log partial likelihood
! Label: 36000~36099
      subroutine coxloglik(no,ni,x,yt,d,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),g(no),w(no),&
      a(ni,nlam),flog(nlam),d(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 36099
36010 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 36019
      jerr=9999
      go to 36099
36019 continue
36020 continue
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) goto 36099
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
36030 do 36039 j=1,ni
      xm(j)=dot_product(q,x(:,j))/sw
36039 continue
36040 do 36049 lam=1,nlam
36041 do 36045 i=1,no
      f(i)=g(i)-gm+dot_product(a(:,lam),(x(i,:)-xm))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
36045 continue
      flog(lam)=risk(no,nk,dq,dk,f,e,kp,jp,uu)
36049 continue
36099 continue
      deallocate(e,uu,dk,f,jp,kp,dq,q,xm)
      end

! baseline hazard
! Label: 36100~36199
      subroutine coxbasehaz(no,ni,x,yt,d,g,w,a,nk,tk,ch,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),g(no),w(no),&
      a(ni),d(no),tk(no),ch(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 36199
36110 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 36119
      jerr=9999
      go to 36199
36119 continue
36120 continue
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) goto 36199
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
36130 do 36139 j=1,ni
      xm(j)=dot_product(q,x(:,j))/sw
36139 continue
      tk(1)=t0
      k=1
36140 do 36149 i=1,no
      f(i)=g(i)-gm+dot_product(a,(x(i,:)-xm))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
      if((dq(i).le.0d0).or.(yt(i).le.t0))goto 36149
      k=k+1
      t0=yt(i)
      tk(k)=t0
36149 continue
      call usk(no,nk,kp,jp,e,uu)
      ch(1:nk)=dk(1:nk)/uu(1:nk)
36199 continue
      deallocate(e,uu,dk,f,jp,kp,dq,q,xm)
      end

! compute the U_i and S_i for nodewise LASSO
! Label: 37000~37999
      subroutine nodewise(no,ni,x,yt,d,g,w,a,ui,si,isi,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),g(no),w(no),&
      a(ni),ui(no,ni),d(no)
      double precision, dimension (:,:) :: si
      double precision, dimension (:), allocatable :: dk,f,dq,q
      double precision, dimension (:), allocatable :: e
      integer, dimension (:), allocatable :: jp,kp
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 37999
37100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 37199
      jerr=9999
      go to 37999
37199 continue
37200 continue
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) goto 37999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
37300 do 37390 j=1,ni
      xm=dot_product(q,x(:,j))/sw
      x(:,j)=x(:,j)-xm
37390 continue
37400 continue
37410 do 37419 i=1,no
      f(i)=g(i)-gm+dot_product(a,x(i,:))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
37419 continue
37500 continue
      if(isi.eq.1)goto 37600
      call nodewiseU(no,ni,nk,kp,jp,x,d,e,ui,jerr)
      goto 37999
37600 continue
      call nodewiseUS(no,ni,nk,kp,jp,x,d,dk,e,ui,si,jerr)
37999 continue
      deallocate(e,dk,f,jp,kp,dq,q)
      end

! compute the U_i's for nodewise LASSO
! Label: 38000~38199
      subroutine nodewiseU(no,ni,nk,kp,jp,x,d,e,ui,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),e(no),ui(no,ni),d(no)
      integer jp(no),kp(nk)
      double precision, dimension (:), allocatable :: h1
      integer, dimension (:), allocatable :: jdk
      allocate(h1(1:ni),stat=jerr)
      allocate(jdk(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)goto 38099
! weighted relative risks
38000 continue
      h0=0.0d0
      h1(:)=0.0d0
38001 do 38090 k=nk,1,-1
      ndk=0
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
38020 do 38029 j=j2,j1,-1
      i=jp(j)
      h0=h0+e(i)
      h1=h1+e(i)*x(i,:)
      if(d(i).le.0.0d0)goto 38029
      ndk=ndk+1
      jdk(ndk)=i
38029 continue
38040 do 38049 j=1,ndk
      i=jdk(j)
      ui(i,:)=x(i,:)-h1/h0
38049 continue
38090 continue
38099 continue
      deallocate(h1,jdk)
      end

! compute the eta_i's for inference
! Label: 38100~38999
      subroutine nodewiseUS(no,ni,nk,kp,jp,x,d,dk,e,ui,si,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),e(no),d(no),dk(nk),&
      ui(no,ni),si(no,ni)
      integer jp(no),kp(nk)
      double precision, dimension (:), allocatable :: hk0,h1
      double precision, dimension (:,:), allocatable :: hk1
      integer, dimension (:), allocatable :: jdk
      allocate(hk1(1:nk,1:ni),stat=jerr)
      allocate(hk0(1:nk),stat=ierr)
      jerr=jerr+ierr
      allocate(h1(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jdk(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)goto 38999
38100 continue
      h0=0.0d0
      h1(:)=0.0d0
38101 do 38190 k=nk,1,-1
      ndk=0
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
38120 do 38129 j=j2,j1,-1
      i=jp(j)
      h0=h0+e(i)
      h1=h1+e(i)*x(i,:)
      if(d(i).le.0.0d0)goto 38129
      ndk=ndk+1
      jdk(ndk)=i
38129 continue
38140 do 38149 j=1,ndk
      i=jdk(j)
      ui(i,:)=x(i,:)-h1/h0
38149 continue
      hk0(k)=h0
      hk1(k,:)=h1
38190 continue
38200 continue
      si=ui
      h0=0.0d0
      h1(:)=0.0d0
38201 do 38290 k=1,nk
      h0=h0+dk(k)/hk0(k)
      h1=h1+(dk(k)/hk0(k)**2)*hk1(k,:)
      j2=kp(k)
      j1=1
      if(k.gt.1)j1=kp(k-1)+1
38210 do 38219 j=j1,j2
      i=jp(j)
      si(i,:)=si(i,:)-(e(i)*h0)*x(i,:)+h1
38219 continue
38290 continue
38999 continue
      deallocate(hk1,hk0,h1,jdk)
      end

! coxLTnet and auxiliaries: Label 40000~49999
! main double precision function: initialization and return
! Label 40000~40999
      subroutine coxLTnet(parm,no,ni,x,yq,yt,d,g,w,jd,vp,cl,ne,nx,nlam,&
     flmin,ulam,thr,  maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer jd(*),ia(nx),nin(nlam)
      double precision, dimension (:), allocatable :: xs,ww,vq
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 40001
      jerr=10000
      return
40001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 40101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
40101 continue
      if(jerr.ne.0) return
      call chkvars(no,ni,x,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 40201
      jerr=7777
      return
40201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 40301
      jerr=9999
      return
40301 continue
      ww=ww/sw
      call cstandard(no,ni,x,ww,ju,isd,xs)
      if(isd .le. 0)goto 40499
40400 do 40401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
40401 continue
40450 continue
40499 continue
      call coxLTnet1(parm,no,ni,x,yq,yt,d,g,ww,ju,vq,cl,ne,nx,nlam,flmin,&
     ulam,thr,  maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 40999
40600 do 40601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
40601 continue
40699 continue
40999 continue
      deallocate(ww,ju,vq)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 41000~43999
      subroutine coxLTnet1(parm,no,ni,x,yq,yt,d,g,q,ju,vp,cl,ne,nx,nlam,&
      flmin,ulam,cthri,  maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),q(no),g(no),vp(ni),ulam(nlam)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ju(ni),m(nx),kin(nlam)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,jq,kq,mm,ixx
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
! jq: orders of truncations times
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
! kq: mark the last index for each groups of unique event times in jq
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 41999
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      if(jerr.ne.0) go to 41999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 41011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 41021
41011 continue
      f=0.0d0
      e=q
41021 continue
41101 continue
      r0=riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
41200 do 41201 i=1,no
!      if((yt(i) .ge. t0) .and. (yq(i) .lt. tk) .and. (q(i) .gt. 0.0d0))goto 41250
      w(i)=0.0d0
      wr(i)=0.0d0
41250 continue
41201 continue
41299 continue
      call outerLT(no,nk,dq,dk,kp,jp,kq,jq,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 41999
      if(flmin .ge. 1.0d0)goto 41301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
41301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
41400 do 41401 j=1,ni
      if(ju(j).eq.0)goto 41401
      ga(j)=abs(dot_product(wr,x(:,j)))
41401 continue
41499 continue
42000 do 42001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 42011
      al=ulam(ilm)
      goto 42099
42011 if(ilm .le. 2)goto 42021
      al=al*alf
      goto 42099
42021 if(ilm .ne. 1)goto 42031
      al=big
      goto 42098
42031 continue
      al0=0.0d0
42100 do 42101 j=1,ni
      if(ju(j).eq.0)goto 42101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
42101 continue
42199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
42098 continue
42099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
42200 do 42201 k=1,ni
      if(ixx(k).eq.1)goto 42201
      if(ju(k).eq.0)goto 42201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
42201 continue
42299 continue
43000 continue
43001 continue
43002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call vars(no,ni,x,w,ixx,v)
! Loop starts
43100 continue
43101 continue
      nlp=nlp+1
      dli=0.0d0
43200 do 43201 j=1,ni
      if(ixx(j).eq.0)goto 43201
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 43211
      at=0.0d0
      goto 43221
43211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
      (v(j)+vp(j)*omal)))
43221 continue
43231 continue
      if(at .eq. a(j))goto 43249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
      if(mm(j) .ne. 0)goto 43248
      nin=nin+1
      if(nin.gt.nx)goto 43299
      mm(j)=nin
      m(nin)=j
43248 continue
43249 continue
43201 continue
43299 continue
      if(nin.gt.nx)goto 43400
      if(dli.lt.cthr)goto 43400
      if(nlp .le. maxit)goto 43300
      jerr=-ilm
      return

43300 continue
43301 continue
43302 continue
      nlp=nlp+1
      dli=0.0d0
43310 do 43311 l=1,nin
      j=m(l)
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 43312
      at=0.0d0
      goto 43313
43312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
43313 continue
43314 continue
      if(at .eq. a(j))goto 43315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
43315 continue
43311 continue
43319 continue
      if(dli.lt.cthr)goto 43331
      if(nlp .le. maxit)goto 43321
      jerr=-ilm
      return
43321 continue
      goto 43302
43331 continue
      goto 43101
43400 continue
      if(nin.gt.nx)goto 43499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outerLT(no,nk,dq,dk,kp,jp,kq,jq,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 43411
      jerr=jerr-ilm
      go to 41999
43411 continue
      ix=0
43420 do 43421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 43421
      ix=1
      goto 43429
43421 continue
43429 continue
      if(ix .ne. 0)goto 43498
43430 do 43431 k=1,ni
      if(ixx(k).eq.1)goto 43431
      if(ju(k).eq.0)goto 43431
      ga(k)=abs(dot_product(wr,x(:,k)))
      if(ga(k) .le. sa*vp(k))goto 43435
      ixx(k)=1
      ix=1
43435 continue
43431 continue
43439 continue
      if(ix.eq.1) go to 43000
      goto 43499
43498 continue
      goto 43002
43499 continue
      if(nin .le. nx)goto 43500
      jerr=-10000-ilm
      goto 42999
43500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)-r0)/rr
      if(ilm.lt.mnl)goto 42001
      if(flmin.ge.1.0d0)goto 42001
      me=0
43510 do 43511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
43511 continue
43999 continue
      if(me.gt.ne)goto 42999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 42999
      if(dev(ilm).gt.devmax)goto 42999
42001 continue
42999 continue
      g=f
41999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,jq,kq,dq,mm,ga,ixx)
      return
      end

! at-risk information
! Label: 44000~44999
     subroutine groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      implicit double precision (a-h,o-z)
      double precision yq(no),yt(no),q(no),d(no)
      integer jp(no),jq(no),kp(*),kq(*)
44010 do 44011 j=1,no
      jp(j)=j
      jq(j)=j
44011 continue
44021 continue
! Get the order of yt, jp
      call psort7(yt,jp,1,no)
      nj=0
! Remove entries with non-positive weights from the order jp
! nj is the number of positive weights
44100 do 44101 j=1,no
      if(q(jp(j)).le.0.0d0)goto 44101
      nj=nj+1
      jp(nj)=jp(j)
44101 continue
44198 continue
      if(nj .ne. 0)goto 44199
! Error if no positive weights
      jerr=20000
      return
44199 continue
! j index for first event time in jp
      j=1
44200 continue
44201 if(d(jp(j)).gt.0.0d0)goto 44209
      j=j+1
      if(j.gt.nj)goto 44209
      goto 44201
44209 continue
      if(j .lt. nj-1)goto 44299
! Error if no event has positive weight
      jerr=30000
      return
44299 continue
! t0 first event time
! j0 index before t0
      t0=yt(jp(j))
      j0=j-1
! j index for last event time in jp
      j=nj
44300 continue
44301 if(d(jp(j)).gt.0.0d0)goto 44309
      j=j-1
!     if(j.le.0)goto 44209 ! not necessary after 44299
      goto 44301
44309 continue
!      if(j .le. 0)goto 44341 ! not necessary after 44320
44320 continue
! tk last event time
! jk index after tk
      tk=yt(jp(j))
      jk=j+1
! Remove entries truncated after the last event time
      if(jk.gt.nj)goto 44399
      nnj=jk-1
44340 do 44341 j=jk,nj
      if(yq(jp(j)).ge.tk)goto 44342
      nnj=nnj+1
      jp(nnj)=jp(j)
44342 continue
44341 continue
      nj=nnj
44399 continue

! Remove entries censored prior to the first event time
! Need to consider ties at t0
      if(j0 .le. 0)goto 44499
44400 continue
44401 if(yt(jp(j0)).lt.t0)goto 44409
      j0=j0-1
      if(j0.eq.0)goto 44409
      goto 44401
44409 continue
      if(j0 .le. 0)goto 44498
      nj=nj-j0
44420 do 44421 j=1,nj
      jp(j)=jp(j+j0)
44421 continue
44429 continue
44498 continue
44499 continue
! sort the truncation times of the remaining entries
      call psort7(yq(jp),jq,1,nj)
      jq=jp(jq)
! nk number of unique event times
! kp mark the last index for each groups of unique event times in jp
! j index in jp
! jj index in jq
      jerr=0
      nk=0
      yk=t0
      j=2
      jj=1
      nnj=nj
44500 continue
44501 if(jj.gt.nnj) goto 44509
      if(yq(jq(jj)).ge.yk) goto 44509
! Remove the entries at risk within consecutive event times
      if(yt(jq(jj)).ge.yk)goto 44511
44520 do 44521 jjj=jj,nnj-1
      jq(jjj)=jq(jjj+1)
44521 continue
      nnj=nnj-1
      goto 44501
44511 continue
      jj=jj+1
      goto 44501
44509 continue
      if(nj.eq.nnj)goto 44599
      jerr=30000
      return
44599 continue
      kq(nk+1)=jj-1
44600 continue
      if(j.le.nj) goto 44601
      nk=nk+1
      kp(nk)=nj
      goto 44999
44601 if((d(jp(j)).gt.0.0d0) .and. (yt(jp(j)).gt.yk))goto 44609
      j=j+1
      if(j.gt.nj)goto 44609
      goto 44601
44609 continue
      nk=nk+1
      kp(nk)=j-1
      if(j.gt.nj)goto 44999
44611 continue
      ykk=yk
      yk=yt(jp(j))
! Search for potential ties with yk sorted to the front
      jjj=j-1
      if(yt(jp(jjj)) .lt. yk)goto 44699
44620 jjj=jjj-1
      if(yt(jp(jjj)) .ge. yk)goto 44620
      kp(nk)=jjj
44699 continue
! Remove the entries at risk within consecutive event times
      jjj=kp(nk-1)+1
44700 if(jjj.ge.j)goto 44799
      if(yq(jp(jjj)).lt.ykk)goto 44795
44720 do 44721 jjjj=jjj,nj-1
      jp(jjjj)=jp(jjjj+1)
44721 continue
      nj=nj-1
      j=j-1
      kp(nk)=kp(nk)-1
      goto 44700
44795 continue
      jjj=jjj+1
      goto 44700
44799 continue
      j=j+1
      goto 44500
44999 continue
      return
      end

! log-likelihood
! Label: no
      double precision function riskLT(no,nk,d,dk,f,e,kp,jp,kq,jq,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),f(no)
      integer kp(nk),jp(no),kq(nk),jq(no)
      double precision e(no),u(nk)
      call uskLT(no,nk,kp,jp,kq,jq,e,u)
      riskLT=dot_product(d,f)-dot_product(dk,log(u))
      return
      end

! total risks at event times
! Label: 45000~45099
      subroutine uskLT(no,nk,kp,jp,kq,jq,e,u)
      implicit double precision (a-h,o-z)
      double precision e(no),u(nk),h
      integer kp(nk),jp(no),kq(nk),jq(no)
      h=0.0d0
45000 do 45090 k=nk,1,-1
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
45020 do 45021 j=j2,j1,-1
      h=h+e(jp(j))
45021 continue
      if(k.ge.nk) goto 45039
      jj2=kq(k+1)
      jj1=kq(k)+1
      do 45031 j=jj2,jj1,-1
      h=h-e(jq(j))
45031 continue
45039 continue
      u(k)=h
45090 continue
45099 continue
      return
      end

! two quantities for coordinate descent
! Label: 45100~45999
      subroutine outerLT(no,nk,d,dk,kp,jp,kq,jq,e,wr,w,jerr,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),wr(no),w(no)
      double precision e(no),u(no),b,c
      integer kp(nk),jp(no),kq(nk),jq(no)
      call uskLT(no,nk,kp,jp,kq,jq,e,u)
      b=dk(1)/u(1)
      c=dk(1)/u(1)**2
      jerr=0
45100 do 45101 j=1,kq(1)
      i=jq(j)
      w(i)=0
      wr(i)=0
45101 continue
45200 do 45201 j=1,kp(1)
      i=jp(j)
      w(i)=e(i)*(b-e(i)*c)
      if(w(i) .gt. 0.0d0)goto 45211
      jerr=-30000
      return
45211 continue
      wr(i)=d(i)-e(i)*b
45201 continue
45299 continue
45300 do 45301 k=2,nk
      jj1=kq(k-1)+1
      jj2=kq(k)
      if(jj1.gt.jj2)goto 45329
45320 do 45321 j=jj1,jj2
      i=jq(j)
      w(i)=-e(i)*(b-e(i)*c)
      wr(i)=e(i)*b
45321 continue
45329 continue
      j1=kp(k-1)+1
      j2=kp(k)
      b=b+dk(k)/u(k)
      c=c+dk(k)/u(k)**2
45340 do 45341 j=j1,j2
      i=jp(j)
      w(i)=w(i)+e(i)*(b-e(i)*c)
      if(w(i) .gt. 0.0d0)goto 45345
      jerr=-30000
      return
45345 continue
      wr(i)=wr(i)+d(i)-e(i)*b
45341 continue
45349 continue
45301 continue
45999 continue
      return
      end

! log partial likelihood
! Label: 46000~46099
      subroutine coxLTloglik(no,ni,x,yq,yt,d,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),g(no),w(no),&
      a(ni,nlam),flog(nlam),d(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp,jq,kq
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 46099
46010 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 46019
      jerr=9999
      go to 46099
46019 continue
46020 continue
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      if(jerr.ne.0) goto 46099
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
46030 do 46039 j=1,ni
      xm(j)=dot_product(q,x(:,j))/sw
46039 continue
46040 do 46049 lam=1,nlam
46041 do 46042 i=1,no
      f(i)=g(i)-gm+dot_product(a(:,lam),(x(i,:)-xm))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
46042 continue
      flog(lam)=riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)
46049 continue
46099 continue
      deallocate(e,uu,dk,f,jp,kp,jq,kq,dq,q,xm)
      end

! baseline hazard
! Label: 46100~46199
      subroutine coxLTbasehaz(no,ni,x,yq,yt,d,g,w,a,nk,tk,ch,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),g(no),w(no),&
      a(ni),tk(no),ch(no),d(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp,jq,kq
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 46199
46110 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 46119
      jerr=9999
      go to 46199
46119 continue
46120 continue
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tkk,jerr)
      if(jerr.ne.0) goto 46199
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
46130 do 46139 j=1,ni
      xm(j)=dot_product(q,x(:,j))/sw
46139 continue
      tk(1)=t0
      k=1
46140 do 46149 i=1,no
      f(i)=g(i)-gm+dot_product(a,(x(i,:)-xm))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
      if((dq(i).le.0d0).or.(yt(i).le.t0))goto 46149
      k=k+1
      t0=yt(i)
      tk(k)=t0
46149 continue
      call uskLT(no,nk,kp,jp,kq,jq,e,uu)
      ch(1:nk)=dk(1:nk)/uu(1:nk)
46199 continue
      deallocate(e,uu,dk,f,jp,kp,jq,kq,dq,q,xm)
      end

! compute the U_i and S_i for nodewise LASSO
! Label: 47000~47999
      subroutine nodewiseLT(no,ni,nid,x,yq,yt,ido,d,g,w,a,ui,si,isi,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),g(no),w(no),&
      a(ni),ui(no,ni),d(no)
      integer ido(no)
      double precision, dimension (:,:) :: si
      double precision, dimension (:), allocatable :: dk,f,dq,q
      double precision, dimension (:), allocatable :: e
      integer, dimension (:), allocatable :: jp,kp,jq,kq,jid
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jid(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 47999
47100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 47199
      jerr=9999
      go to 47999
47199 continue
47200 continue
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      if(jerr.ne.0) goto 47999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
47300 do 47390 j=1,ni
      xm=dot_product(q,x(:,j))/sw
      x(:,j)=x(:,j)-xm
47390 continue
47400 continue
47410 do 47419 i=1,no
      f(i)=g(i)-gm+dot_product(a,x(i,:))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
47419 continue
47500 continue
      if(isi.eq.1)goto 47510
      call nodewiseULT(no,ni,nk,kp,jp,kq,jq,x,d,e,ui,jerr)
      goto 47599
47510 continue
      call nodewiseUSLT(no,ni,nk,kp,jp,kq,jq,x,d,dk,e,ui,si,jerr)
47520 continue
47531 do 47538 i=1,no
      jid(i)=i
47538 continue
      call psort7int(ido,jid,1,no)
47560 continue
      nid=1
      idj=jid(1)
      j=1
47561 continue
      j=j+1
      if(j.gt.no) goto 47590
      if(ido(jid(j)).gt.ido(idj))goto 47580
      si(idj,:)=si(idj,:)+si(jid(j),:)
      si(jid(j),:)=0d0
      goto 47561
47580 continue
      nid=nid+1
      idj=jid(j)
      goto 47561
47590 continue
47599 continue
47999 continue
      deallocate(e,dk,f,jp,kp,jq,kq,dq,q,jid)
      end

! compute the U_i's for nodewise LASSO
! Label: 48000~48199
      subroutine nodewiseULT(no,ni,nk,kp,jp,kq,jq,x,d,e,ui,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),e(no),ui(no,ni),d(no)
      integer jp(no),kp(nk),jq(no),kq(nk)
      double precision, dimension (:), allocatable :: h1,hk1
      allocate(h1(1:ni),stat=jerr)
      allocate(hk1(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)goto 48099
! weighted relative risks
48000 continue
      h0=0.0d0
      h1(:)=0.0d0
48001 do 48090 k=1,nk
      j2=kq(k)
      j1=1
      if(k.gt.1)j1=kq(k-1)+1
      if(j1.gt.j2)goto 48020
48010 do 48019 j=j1,j2
      i=jq(j)
      h0=h0+e(i)
      h1=h1+e(i)*x(i,:)
48019 continue
      hk0=h0
      hk1=h1
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
48020 do 48029 j=j1,j2
      i=jp(j)
      h0=h0-e(i)
      h1=h1-e(i)*x(i,:)
      if(d(i).le.0.0d0)goto 48029
      ui(i,:)=x(i,:)-hk1/hk0
48029 continue
48090 continue
48099 continue
      deallocate(h1,hk1)
      end

! compute the eta_i's for inference
! Label: 48100~48999
      subroutine nodewiseUSLT(no,ni,nk,kp,jp,kq,jq,x,d,dk,e,ui,si,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),e(no),d(no),dk(nk),&
      ui(no,ni),si(no,ni)
      integer jp(no),kp(nk),jq(no),kq(nk)
      double precision, dimension (:), allocatable :: hk0,h1
      double precision, dimension (:,:), allocatable :: hk1
      allocate(hk1(1:nk,1:ni),stat=jerr)
      allocate(hk0(1:nk),stat=ierr)
      jerr=jerr+ierr
      allocate(h1(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)goto 48999
48100 continue
      h0=0.0d0
      h1(:)=0.0d0
48101 do 48190 k=1,nk
      j2=kq(k)
      j1=1
      if(k.gt.1)j1=kq(k-1)+1
      if(j1.gt.j2)goto 48120
48110 do 48119 j=j1,j2
      i=jq(j)
      h0=h0+e(i)
      h1=h1+e(i)*x(i,:)
48119 continue
      hk0(k)=h0
      hk1(k,:)=h1
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
48120 do 48129 j=j1,j2
      i=jp(j)
      h0=h0-e(i)
      h1=h1-e(i)*x(i,:)
      if(d(i).le.0.0d0)goto 48129
      ui(i,:)=x(i,:)-hk1(k,:)/hk0(k)
48129 continue
48190 continue
48200 continue
      si=ui
      h0=0.0d0
      h1(:)=0.0d0
48201 do 48290 k=1,nk
      j2=kq(k)
      j1=1
      if(k.gt.1)j1=kq(k-1)+1
      if(j1.gt.j2)goto 48219
48210 do 48218 j=j1,j2
      i=jq(j)
      si(i,:)=si(i,:)+(e(i)*h0)*x(i,:)-h1
48218 continue
48219 continue
      h0=h0+dk(k)/hk0(k)
      h1=h1+(dk(k)/hk0(k)**2)*hk1(k,:)
      j2=kp(k)
      j1=1
      if(k.gt.1)j1=kp(k-1)+1
48230 do 48239 j=j1,j2
      i=jp(j)
      si(i,:)=si(i,:)-(e(i)*h0)*x(i,:)+h1
48239 continue
48290 continue
48999 continue
      deallocate(hk1,hk0,h1)
      end

! spcoxnet and auxiliaries: Label 50000~53999
! main double precision function: initialization and return
! Label 50000~50999
      subroutine spcoxnet(parm,no,ni,x,ix,jx,yt,d,g,w,jd,vp,cl,ne,nx,&
  nlam,flmin,ulam,thr,  maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)
      double precision, dimension (:), allocatable :: xs,ww,vq,xm
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 50001
      jerr=10000
      return
50001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 50101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
50101 continue
      if(jerr.ne.0) return
      call spchkvars(no,ni,x,ix,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 50201
      jerr=7777
      return
50201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 50301
      jerr=9999
      return
50301 continue
      ww=ww/sw
      call spcstandard(no,ni,x,ix,jx,ww,ju,isd,xs,xm)
      if(isd .le. 0)goto 50499
50400 do 50401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
50401 continue
50450 continue
50499 continue
      call spcoxnet1(parm,no,ni,x,ix,jx,xm,yt,d,g,ww,ju,vq,cl,ne,nx,&
nlam,flmin,ulam,thr,maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 50999
50600 do 50601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
50601 continue
50699 continue
50999 continue
      deallocate(ww,ju,vq,xm)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 51000~53999
      subroutine spcoxnet1(parm,no,ni,x,ix,jx,xm,yt,d,g,q,ju,vp,cl,ne,nx,&
nlam,flmin,ulam,cthri,  maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),q(no),g(no),vp(ni),ulam(nlam),xm(ni)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,mm,ixx
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 51999
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) go to 51999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 51011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 51021
51011 continue
      f=0.0d0
      e=q
51021 continue
51101 continue
      r0=risk(no,nk,dq,dk,f,e,kp,jp,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
51200 do 51201 i=1,no
!      if((yt(i) .ge. t0) .and. (q(i) .gt. 0.0d0))goto 51250
      w(i)=0.0d0
      wr(i)=w(i)
51250 continue
51201 continue
51299 continue
      call outer(no,nk,dq,dk,kp,jp,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 51999
      if(flmin .ge. 1.0d0)goto 51301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
51301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
51400 wrtot = sum(wr)
      do 51401 j=1,ni
      if(ju(j).eq.0)goto 51401
      jb=ix(j)
      je=ix(j+1)-1
      ga(j)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j))
51401 continue
51499 continue
52000 do 52001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 52011
      al=ulam(ilm)
      goto 52099
52011 if(ilm .le. 2)goto 52021
      al=al*alf
      goto 52099
52021 if(ilm .ne. 1)goto 52031
      al=big
      goto 52098
52031 continue
      al0=0.0d0
52100 do 52101 j=1,ni
      if(ju(j).eq.0)goto 52101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
52101 continue
52199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
52098 continue
52099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
52200 do 52201 k=1,ni
      if(ixx(k).eq.1)goto 52201
      if(ju(k).eq.0)goto 52201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
52201 continue
52299 continue
53000 continue
53001 continue
53002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call spvars(no,ni,x,ix,jx,xm,w,ixx,v)

! Loop starts
53100 continue
53101 continue
      nlp=nlp+1
      dli=0.0d0
53200 wrtot = sum(wr)
      do 53201 j=1,ni
      if(ixx(j).eq.0)goto 53201
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 53211
      at=0.0d0
      goto 53221
53211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
53221 continue
53231 continue
      if(at .eq. a(j))goto 53249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f = f - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
      if(mm(j) .ne. 0)goto 53248
      nin=nin+1
      if(nin.gt.nx)goto 53299
      mm(j)=nin
      m(nin)=j
53248 continue
53249 continue
53201 continue
53299 continue
      if(nin.gt.nx)goto 53400
      if(dli.lt.cthr)goto 53400
      if(nlp .le. maxit)goto 53300
      jerr=-ilm
      return

53300 continue
53301 continue
53302 continue
      nlp=nlp+1
      dli=0.0d0
53310 wrtot = sum(wr)
      do 53311 l=1,nin
      j=m(l)
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 53312
      at=0.0d0
      goto 53313
53312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
53313 continue
53314 continue
      if(at .eq. a(j))goto 53315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f = f - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
53315 continue
53311 continue
53319 continue
      if(dli.lt.cthr)goto 53331
      if(nlp .le. maxit)goto 53321
      jerr=-ilm
      return
53321 continue
      goto 53302
53331 continue
      goto 53101
53400 continue
      if(nin.gt.nx)goto 53499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outer(no,nk,dq,dk,kp,jp,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 53411
      jerr=jerr-ilm
      go to 51999
53411 continue
      indx=0
53420 do 53421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 53421
      indx=1
      goto 53429
53421 continue
53429 continue
      if(indx .ne. 0)goto 53498
53430 wrtot = sum(wr)
      do 53431 k=1,ni
      if(ixx(k).eq.1)goto 53431
      if(ju(k).eq.0)goto 53431
      jb=ix(k)
      je=ix(k+1)-1
      ga(k)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(k))
      if(ga(k) .le. sa*vp(k))goto 53435
      ixx(k)=1
      indx=1
53435 continue
53431 continue
53439 continue
      if(indx.eq.1) go to 53000
      goto 53499
53498 continue
      goto 53002
53499 continue
      if(nin .le. nx)goto 53500
      jerr=-10000-ilm
      goto 52999
53500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(risk(no,nk,dq,dk,f,e,kp,jp,uu)-r0)/rr
      if(ilm.lt.mnl)goto 52001
      if(flmin.ge.1.0d0)goto 52001
      me=0
53510 do 53511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
53511 continue
53999 continue
      if(me.gt.ne)goto 52999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 52999
      if(dev(ilm).gt.devmax)goto 52999
52001 continue
52999 continue
      g=f
51999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx)
      return
      end

! spcoxLTnet and auxiliaries: Label 54000~57999
! main double precision function: initialization and return
! Label 54000~54999
      subroutine spcoxLTnet(parm,no,ni,x,ix,jx,yq,yt,d,g,w,jd,vp,cl,ne,&
nx,nlam,flmin,ulam,thr,  maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam)
      double precision, dimension (:), allocatable :: xs,ww,vq,xm
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 54001
      jerr=10000
      return
54001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 54101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
54101 continue
      if(jerr.ne.0) return
      call spchkvars(no,ni,x,ix,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 54201
      jerr=7777
      return
54201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 54301
      jerr=9999
      return
54301 continue
      ww=ww/sw
      call spcstandard(no,ni,x,ix,jx,ww,ju,isd,xs,xm)
      if(isd .le. 0)goto 54499
54400 do 54401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
54401 continue
54450 continue
54499 continue
      call spcoxLTnet1(parm,no,ni,x,ix,jx,xm,yq,yt,d,g,ww,ju,vq,cl,ne,nx,&
nlam,flmin,ulam,thr,maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 54999
54600 do 54601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
54601 continue
54699 continue
54999 continue
      deallocate(ww,ju,vq,xm)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 55000~57999
      subroutine spcoxLTnet1(parm,no,ni,x,ix,jx,xm,yq,yt,d,g,q,ju,vp,cl,ne,&
nx,nlam,flmin,ulam,cthri,  maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),q(no),g(no),vp(ni),ulam(nlam),xm(ni)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,jq,kq,mm,ixx
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
! jq: orders of truncations times
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
! kq: mark the last index for each groups of unique event times in jq
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 55999
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      if(jerr.ne.0) go to 55999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 55011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 55021
55011 continue
      f=0.0d0
      e=q
55021 continue
55101 continue
      r0=riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
55200 do 55201 i=1,no
!      if((yt(i) .ge. t0) .and. (yq(i) .lt. tk) .and. (q(i) .gt. 0.0d0))goto 55250
      w(i)=0.0d0
      wr(i)=w(i)
55250 continue
55201 continue
55299 continue
      call outerLT(no,nk,dq,dk,kp,jp,kq,jq,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 55999
      if(flmin .ge. 1.0d0)goto 55301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
55301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
55400 wrtot = sum(wr)
      do 55401 j=1,ni
      if(ju(j).eq.0)goto 55401
      jb=ix(j)
      je=ix(j+1)-1
      ga(j)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j))
55401 continue
55499 continue
56000 do 56001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 56011
      al=ulam(ilm)
      goto 56099
56011 if(ilm .le. 2)goto 56021
      al=al*alf
      goto 56099
56021 if(ilm .ne. 1)goto 56031
      al=big
      goto 56098
56031 continue
      al0=0.0d0
56100 do 56101 j=1,ni
      if(ju(j).eq.0)goto 56101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
56101 continue
56199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
56098 continue
56099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
56200 do 56201 k=1,ni
      if(ixx(k).eq.1)goto 56201
      if(ju(k).eq.0)goto 56201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
56201 continue
56299 continue
57000 continue
57001 continue
57002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call spvars(no,ni,x,ix,jx,xm,w,ixx,v)

! Loop starts
57100 continue
57101 continue
      nlp=nlp+1
      dli=0.0d0
57200 wrtot = sum(wr)
      do 57201 j=1,ni
      if(ixx(j).eq.0)goto 57201
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 57211
      at=0.0d0
      goto 57221
57211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
57221 continue
57231 continue
      if(at .eq. a(j))goto 57249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
      if(mm(j) .ne. 0)goto 57248
      nin=nin+1
      if(nin.gt.nx)goto 57299
      mm(j)=nin
      m(nin)=j
57248 continue
57249 continue
57201 continue
57299 continue
      if(nin.gt.nx)goto 57400
      if(dli.lt.cthr)goto 57400
      if(nlp .le. maxit)goto 57300
      jerr=-ilm
      return

57300 continue
57301 continue
57302 continue
      nlp=nlp+1
      dli=0.0d0
57310 wrtot = sum(wr)
      do 57311 l=1,nin
      j=m(l)
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 57312
      at=0.0d0
      goto 57313
57312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
57313 continue
57314 continue
      if(at .eq. a(j))goto 57315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
57315 continue
57311 continue
57319 continue
      if(dli.lt.cthr)goto 57331
      if(nlp .le. maxit)goto 57321
      jerr=-ilm
      return
57321 continue
      goto 57302
57331 continue
      goto 57101
57400 continue
      if(nin.gt.nx)goto 57499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outerLT(no,nk,dq,dk,kp,jp,kq,jq,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 57411
      jerr=jerr-ilm
      go to 55999
57411 continue
      indx=0
57420 do 57421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 57421
      indx=1
      goto 57429
57421 continue
57429 continue
      if(indx .ne. 0)goto 57498
57430 wrtot = sum(wr)
      do 57431 k=1,ni
      if(ixx(k).eq.1)goto 57431
      if(ju(k).eq.0)goto 57431
      jb=ix(k)
      je=ix(k+1)-1
      ga(k)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(k))
      if(ga(k) .le. sa*vp(k))goto 57435
      ixx(k)=1
      indx=1
57435 continue
57431 continue
57439 continue
      if(indx.eq.1) go to 57000
      goto 57499
57498 continue
      goto 57002
57499 continue
      if(nin .le. nx)goto 57500
      jerr=-10000-ilm
      goto 56999
57500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)-r0)/rr
      if(ilm.lt.mnl)goto 56001
      if(flmin.ge.1.0d0)goto 56001
      me=0
57510 do 57511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
57511 continue
57999 continue
      if(me.gt.ne)goto 56999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 56999
      if(dev(ilm).gt.devmax)goto 56999
56001 continue
56999 continue
      g=f
55999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,jq,kq,dq,mm,ga,ixx)
      return
      end

! log partial likelihood
! Label: 58000~58099
      subroutine spcoxloglik(no,ni,x,ix,jx,yt,d,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),g(no),w(no),a(ni,nlam),flog(nlam),d(no)
      integer ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 58099
58010 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 58019
      jerr=9999
      go to 58099
58019 continue
58020 continue
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) goto 58099
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
58030 do 58039 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
58039 continue
58040 continue
      f=g-gm
58041 do 58042 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 58042
      f  = f  - a(j,1)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j,1)*x(jb:je)
58042 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(1)=risk(no,nk,dq,dk,f,e,kp,jp,uu)
58050 if(nlam.lt.2) goto 58099
58051 do 58059 lam=2,nlam
58052 do 58053 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 58053
      del = a(j,lam)-a(j,lam-1)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
58053 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(lam)=risk(no,nk,dq,dk,f,e,kp,jp,uu)
58059 continue
58099 continue
      deallocate(e,uu,dk,f,jp,kp,dq,xm,q)
      end

! baseline hazard
! Label: 58100~58199
      subroutine spcoxbasehaz(no,ni,x,ix,jx,yt,d,g,w,a,nk,tk,ch,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),g(no),w(no),a(ni),d(no),tk(no),ch(no)
      integer ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 58199
58110 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 58119
      jerr=9999
      go to 58199
58119 continue
58120 continue
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) goto 58199
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
58130 do 58139 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
58139 continue
58140 continue
      f=g-gm
58141 do 58142 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 58142
      f  = f  - a(j)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j)*x(jb:je)
58142 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      call usk(no,nk,kp,jp,e,uu)
      ch(1:nk)=dk(1:nk)/uu(1:nk)
      tk(1)=t0
58150 do 58159 k=2,nk
      jb=kp(k-1)
      je=kp(k)
58151 do 58152 j=jb,je
      i=jp(j)
      if(dq(i).gt.0d0)goto 58153
58152 continue
58153 continue
      tk(k)=yt(i)
58159 continue
58199 continue
      deallocate(e,uu,dk,f,jp,kp,dq,xm,q)
      end

! log partial likelihood
! Label: 59000~59099
      subroutine spcoxLTloglik(no,ni,x,ix,jx,yq,yt,d,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),g(no),w(no),a(ni,nlam),flog(nlam),d(no)
      integer ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp,jq,kq
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 59099
59010 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 59019
      jerr=9999
      go to 59099
59019 continue
59020 continue
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      if(jerr.ne.0) goto 59099
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
59030 do 59039 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
59039 continue
59040 continue
      f=g-gm
59041 do 59042 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 59042
      f  = f  - a(j,1)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j,1)*x(jb:je)
59042 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(1)=riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)
59050 if(nlam.lt.2) goto 59099
59051 do 59059 lam=2,nlam
59052 do 59053 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 59053
      del = a(j,lam)-a(j,lam-1)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
59053 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(lam)=riskLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,uu)
59059 continue
59099 continue
      deallocate(e,uu,dk,f,jp,kp,dq,xm,q,jq,kq)
      end

! baseline hazard
! Label: 59100~59199
      subroutine spcoxLTbasehaz(no,ni,x,ix,jx,yq,yt,d,g,w,a,nk,tk,ch,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),g(no),w(no),a(ni),d(no),tk(no),ch(no)
      integer ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu
      integer, dimension (:), allocatable :: jp,kp,jq,kq
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 59199
59110 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 59119
      jerr=9999
      go to 59199
59119 continue
59120 continue
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tkk,jerr)
      if(jerr.ne.0) goto 59199
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
59130 do 59139 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
59139 continue
59140 continue
      f=g-gm
59141 do 59142 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 59142
      f  = f  - a(j)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j)*x(jb:je)
59142 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      call uskLT(no,nk,kp,jp,kq,jq,e,uu)
      ch(1:nk)=dk(1:nk)/uu(1:nk)
      tk(1)=t0
59150 do 59159 k=2,nk
      jb=kp(k-1)
      je=kp(k)
59151 do 59152 j=jb,je
      i=jp(j)
      if(dq(i).gt.0d0)goto 59153
59152 continue
59153 continue
      tk(k)=yt(i)
59159 continue
59199 continue
      deallocate(e,uu,dk,f,jp,kp,dq,xm,q,jq,kq)
      end



! fgnet and auxiliaries: Label 60000~69999
! main double precision function: initialization and return
! Label 60000~60999
      subroutine fgnet(parm,no,ni,x,yt,d,cr,g,w,jd,vp,cl,ne,&
nx,nlam,flmin,ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer jd(*),ia(nx),nin(nlam),cr(no)
      double precision, dimension (:), allocatable :: xs,ww,vq
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 60001
      jerr=10000
      return
60001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 60101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
60101 continue
      if(jerr.ne.0) return
      call chkvars(no,ni,x,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 60201
      jerr=7777
      return
60201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 60301
      jerr=9999
      return
60301 continue
      ww=ww/sw
      call cstandard(no,ni,x,ww,ju,isd,xs)
      if(isd .le. 0)goto 60499
60400 do 60401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
60401 continue
60450 continue
60499 continue
      call fgnet1(parm,no,ni,x,yt,d,cr,g,ww,ju,vq,cl,ne,nx,&
nlam,flmin,ulam,thr,maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 60999
60600 do 60601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
60601 continue
60699 continue
60999 continue
      deallocate(ww,ju,vq)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 61000~63999
      subroutine fgnet1(parm,no,ni,x,yt,d,cr,g,q,ju,vp,cl,ne,nx,&
nlam,flmin,ulam,cthri,maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),q(no),g(no),vp(ni),ulam(nlam)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ju(ni),m(nx),kin(nlam),cr(no)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq,km
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,jcr,kcr,mm,ixx
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 61999
      call groupsFG(no,yt,cr,q,nk,kp,jp,kcr,jcr,km,t0,jerr)
      if(jerr.ne.0) go to 61999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 61011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 61021
61011 continue
      f=0.0d0
      e=q
61021 continue
61101 continue
      r0=riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
61200 do 61201 i=1,no
      w(i)=0.0d0
      wr(i)=w(i)
61250 continue
61201 continue
61299 continue
      call outerFG(no,nk,dq,dk,kp,jp,kcr,jcr,km,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 61999
      if(flmin .ge. 1.0d0)goto 61301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
61301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
61400 do 61401 j=1,ni
      if(ju(j).eq.0)goto 61401
      ga(j)=abs(dot_product(wr,x(:,j)))
61401 continue
61499 continue
62000 do 62001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 62011
      al=ulam(ilm)
      goto 62099
62011 if(ilm .le. 2)goto 62021
      al=al*alf
      goto 62099
62021 if(ilm .ne. 1)goto 62031
      al=big
      goto 62098
62031 continue
      al0=0.0d0
62100 do 62101 j=1,ni
      if(ju(j).eq.0)goto 62101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
62101 continue
62199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
62098 continue
62099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
62200 do 62201 k=1,ni
      if(ixx(k).eq.1)goto 62201
      if(ju(k).eq.0)goto 62201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
62201 continue
62299 continue
63000 continue
63001 continue
63002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call vars(no,ni,x,w,ixx,v)

! Loop starts
63100 continue
63101 continue
      nlp=nlp+1
      dli=0.0d0
63200 do 63201 j=1,ni
      if(ixx(j).eq.0)goto 63201
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 63211
      at=0.0d0
      goto 63221
63211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
63221 continue
63231 continue
      if(at .eq. a(j))goto 63249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
      if(mm(j) .ne. 0)goto 63248
      nin=nin+1
      if(nin.gt.nx)goto 63299
      mm(j)=nin
      m(nin)=j
63248 continue
63249 continue
63201 continue
63299 continue
      if(nin.gt.nx)goto 63400
      if(dli.lt.cthr)goto 63400
      if(nlp .le. maxit)goto 63300
      jerr=-ilm
      return

63300 continue
63301 continue
63302 continue
      nlp=nlp+1
      dli=0.0d0
63310 do 63311 l=1,nin
      j=m(l)
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 63312
      at=0.0d0
      goto 63313
63312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
63313 continue
63314 continue
      if(at .eq. a(j))goto 63315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
63315 continue
63311 continue
63319 continue
      if(dli.lt.cthr)goto 63331
      if(nlp .le. maxit)goto 63321
      jerr=-ilm
      return
63321 continue
      goto 63302
63331 continue
      goto 63101
63400 continue
      if(nin.gt.nx)goto 63499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outerFG(no,nk,dq,dk,kp,jp,kcr,jcr,km,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 63411
      jerr=jerr-ilm
      go to 61999
63411 continue
      ix=0
63420 do 63421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 63421
      ix=1
      goto 63429
63421 continue
63429 continue
      if(ix .ne. 0)goto 63498
63430 do 63431 k=1,ni
      if(ixx(k).eq.1)goto 63431
      if(ju(k).eq.0)goto 63431
      ga(k)=abs(dot_product(wr,x(:,k)))
      if(ga(k) .le. sa*vp(k))goto 63435
      ixx(k)=1
      ix=1
63435 continue
63431 continue
63439 continue
      if(ix.eq.1) go to 63000
      goto 63499
63498 continue
      goto 63002
63499 continue
      if(nin .le. nx)goto 63500
      jerr=-10000-ilm
      goto 62999
63500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)-r0)/rr
      if(ilm.lt.mnl)goto 62001
      if(flmin.ge.1.0d0)goto 62001
      me=0
63510 do 63511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
63511 continue
63999 continue
      if(me.gt.ne)goto 62999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 62999
      if(dev(ilm).gt.devmax)goto 62999
62001 continue
62999 continue
      g=f
61999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx,km,kcr,jcr)
      return
      end

! get size for IPW at-risk information
! Label 64000~64999
      subroutine groupsFG(no,yt,cr,q,nk,kp,jp,kcr,jcr,km,t0,jerr)
      implicit double precision (a-h,o-z)
      double precision yt(no),q(no),km(no)
      integer jp(no),cr(no),jcr(no),kp(*),kcr(*)
      integer, dimension (:), allocatable :: jpa
      allocate(jpa(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 64999

64010 do 64011 j=1,no
      jpa(j)=j
64011 continue
64021 continue
! Get the order of yt, jpa
      call psort7(yt,jpa,1,no)
! Remove entries with non-positive weights from the order jpa
! nj is the number of positive weights
      nj=0
64030 do 64031 j=1,no
      if(q(jpa(j)).le.0.0d0)goto 64031
      nj=nj+1
      jpa(nj)=jpa(j)
64031 continue
64039 continue
      if(nj .ne. 0)goto 64099
! Error if no positive weights
      jerr=20000
      return
64099 continue
! Kaplan-Meier
      gj=1
      j=1
64100 continue
64101 if(j.gt.nj)goto 64199
64102 if(cr(jpa(j)).gt. 0)goto 64150
      yb=yt(jpa(j))
      qc=q(jpa(j))
      jb=j
64110 continue
64111 if(jb.eq.1)goto 64119
64112 if(yt(jpa(jb-1)).lt.yb)goto 64119
      jb=jb-1
      goto 64111
64119 continue
      qar=sum(q(jpa(jb:nj)))
64120 continue
      j=j+1
64121 if(j.gt.nj)goto 64129
64122 if(yt(jpa(j)).gt.yb)goto 64129
64130 continue
64131 if(cr(jpa(j)).gt.0) goto 64139
      qc=qc+q(jpa(j))
64139 continue
      goto 64120
64129 continue
64140 continue
      je=j-1
      gj=gj*(1-qc/qar)
      km(jpa(jb:je))=gj
      goto 64101
64150 continue
      km(jpa(j))=gj
      j=j+1
      goto 64101
64199 continue
! deal with censoring/CR events prior to first type-1 event
! ncr number of competing risks events
      j=1
      ncr=0
      kpj=1
64200 continue
64201 if(cr(jpa(j)).eq.1)goto 64230
64211 if(cr(jpa(j)).eq.0)goto 64219
      ncr=ncr+1
      jcr(ncr)=jpa(j)
64219 continue
      j=j+1
      goto 64201
64230 continue
      jp(1)=jpa(j)
      t0=yt(jpa(j))
      j0=j-1
64231 if(j0.eq.0)goto 64239
64232 if(yt(jpa(j0)).lt.t0)goto 64239
      if(cr(jpa(j0)).gt.1)goto 64235
      kpj=kpj+1
      jp(kpj)=j0
64235 continue
      j0=j0-1
      goto 64231
64239 continue
64240 continue
      gj = km(jp(1))
      kcr(1)=ncr
64241 if(kcr(1).eq. 0) goto 64299
      if(km(jcr(kcr(1))).gt. gj) goto 64299
      kcr(1)=kcr(1)-1
      goto 64241
64249 continue
64299 continue
! nk number of unique event times
! kp mark the last index for each groups of unique event times in jp
! j index in jp
! jj index in jq
64300 continue
      jerr=0
      nk=0
      yk=t0
64301 j=j+1
64302 if(j.gt.nj)goto 64399
64311 if(cr(jpa(j)).eq.2)goto 64330
      kpj=kpj+1
      jp(kpj)=jpa(j)
64321 if(yt(jpa(j)).gt.yk .and. cr(jpa(j)).eq.1)goto 64340
      goto 64301
64330 continue
      ncr=ncr+1
      jcr(ncr)=jpa(j)
      goto 64301
64340 continue
      nk=nk+1
      yk=yt(jpa(j))
! potential ties with yk to the front
      kpjj=kpj-1
64341 if(yt(jp(kpjj)).lt.yk) goto 64348
      kpjj = kpjj - 1
      goto 64341
64348 continue
      kp(nk)=kpjj
64360 continue
      gj = km(jp(kpj))
      kcrjj=ncr
64361 if(kcrjj .eq. 0)goto 64368
      if(km(jcr(kcrjj)).gt. gj) goto 64368
      kcrjj = kcrjj - 1
      goto 64361
64368 continue
      kcr(nk+1)=kcrjj
      goto 64301
64399 continue
      nk=nk+1
      kp(nk)=kpj
      kcr(nk+1) = ncr
64999 continue
      deallocate(jpa)
      return
      end

! log-likelihood
! Label: 65000~65999
      double precision function riskFG(no,nk,d,dk,f,e,kp,jp,kcr,jcr,km,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),f(no),km(no)
      integer kp(nk),jp(no),kcr(nk+1),jcr(no)
      double precision e(no),u(nk)
      call uskFG(no,nk,kp,jp,kcr,jcr,km,e,u)
      riskFG=dot_product(d,f)-dot_product(dk,log(u))
      return
      end

! total risks at event times
! Label: 66000~66999
      subroutine uskFG(no,nk,kp,jp,kcr,jcr,km,e,u)
      implicit double precision (a-h,o-z)
      double precision e(no),u(nk),km(no)
      integer kp(nk),jp(no),kcr(nk+1),jcr(no)
      h=0.0d0
      hG=0.0d0
66100 do 66190 k=1,(nk+1)
      jj2=kcr(k)
      jj1=1
      if(k .gt. 1) jj1=kcr(k-1)+1
66110 if(jj2 .lt. jj1) goto 66180
66120 do 66129 j=jj1,jj2
      hG = hG + e(jcr(j))/km(jcr(j))
66129 continue
66180 continue
66190 continue
66200 do 66990 k=nk,1,-1
66300 continue
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
66321 do 66329 j=j2,j1,-1
      h=h+e(jp(j))
66329 continue
66400 continue
      g1=km(jp(j1))
      g2=0
      if(k.lt.nk) g2=km(jp(j2+1))
66500 continue
      jj2=kcr(k+1)
      jj1=kcr(k)+1
66501 if(jj2 .lt. jj1) goto 66599
66510 do 66519 j=jj1,jj2
      hG = hG - e(jcr(j))/km(jcr(j))
      h = h + e(jcr(j))*(1-g2/km(jcr(j)))
66519 continue
66599 continue
      h=h+hG*(g1-g2)
      u(k)=h
66990 continue
66999 continue
      return
      end

! two quantities for coordinate descent
! Label: 67000~67999
      subroutine outerFG(no,nk,d,dk,kp,jp,kcr,jcr,km,e,wr,w,jerr,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),wr(no),w(no),km(no)
      double precision e(no),u(no),b,c
      integer kp(nk),jp(no),kcr(nk+1),jcr(no)
      call uskFG(no,nk,kp,jp,kcr,jcr,km,e,u)
      b=dk(1)/u(1)
      bG=b*km(jp(1))
      c=dk(1)/u(1)**2
      cG=c*km(jp(1))**2
      jerr=0
67100 do 67190 j=1,kp(1)
      i=jp(j)
      w(i)=e(i)*(b-e(i)*c)
      if(w(i) .gt. 0.0d0)goto 67111
      jerr=-30000
      return
67111 continue
      wr(i)=d(i)-e(i)*b
67190 continue
67200 continue
      j1=kcr(1)+1
      j2=kcr(2)
67201 if(j1.gt.j2) goto 67299
67202 do 67290 j=j1,j2
      i = jcr(j)
      w(i)=e(i)*(b-e(i)*c)- e(i)/km(i)*(bG-e(i)/km(i)*cG)
      wr(i)=-e(i)*b+e(i)/km(i)*bG
67290 continue
67299 continue
67500 continue
67501 do 67890 k=2,nk
      j1=kp(k-1)+1
      gj=km(jp(j1))
      j2=kp(k)
      bk=dk(k)/u(k)
      ck=dk(k)/u(k)**2
      b=b+bk
      c=c+ck
      bG=bG+bk*gj
      cG=cG+ck*gj**2
67600 do 67690 j=j1,j2
      i=jp(j)
      w(i)=e(i)*(b-e(i)*c)
      if(w(i) .gt. 0.0d0)goto 67611
      jerr=-30000
      return
67611 continue
      wr(i)=d(i)-e(i)*b
67690 continue

67700 continue
      j1=kcr(k)+1
      j2=kcr(k+1)
67701 if(j1.gt.j2) goto 67799
67702 do 67790 j=j1,j2
      i = jcr(j)
      w(i)=e(i)*(b-e(i)*c)- e(i)/km(i)*(bG-e(i)/km(i)*cG)
      wr(i)=-e(i)*b+e(i)/km(i)*bG
67790 continue
67799 continue
67890 continue
67900 continue
67901 do 67990 j=1,kcr(nk+1)
      i = jcr(j)
      w(i)=w(i)+e(i)/km(i)*(bG-e(i)/km(i)*cG)
      wr(i)=wr(i)-e(i)/km(i)*bG
67990 continue
67999 continue
      return
      end

! log partial likelihood
! Label: 68000~68999
      subroutine fgloglik(no,ni,x,yt,d,cr,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yt(no),d(no),g(no),w(no),a(ni,nlam),flog(nlam)
      integer cr(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu,km
      integer, dimension (:), allocatable :: jp,kp,jcr,kcr
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 68999
68100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 68199
      jerr=9999
      go to 68999
68199 continue
68200 continue
      call groupsFG(no,yt,cr,q,nk,kp,jp,kcr,jcr,km,t0,jerr)
      if(jerr.ne.0) goto 68999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
68300 do 68390 j=1,ni
      xm(j)=dot_product(q,x(:,j))/sw
68390 continue
68400 do 68490 lam=1,nlam
68410 do 68419 i=1,no
      f(i)=g(i)-gm+dot_product(a(:,lam),(x(i,:)-xm))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
68419 continue
      flog(lam)=riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)
68490 continue
      deallocate(e,uu,dk,f,jp,kp,dq,kcr,jcr,km,xm,q)
68999 continue
      end


! fgLTnet and auxiliaries: Label 70000~79999
! main double precision function: initialization and return
! Label 70000~70999
      subroutine fgLTnet(parm,no,ni,x,yq,yt,ycr,d,cr,g,w,jd,vp,cl,&
ne,nx,nlam,flmin,ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),ycr(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer jd(*),ia(nx),nin(nlam),cr(no)
      double precision, dimension (:), allocatable :: xs,ww,vq
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 70001
      jerr=10000
      return
70001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 70101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
70101 continue
      if(jerr.ne.0) return
      call chkvars(no,ni,x,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 70201
      jerr=7777
      return
70201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 70301
      jerr=9999
      return
70301 continue
      ww=ww/sw
      call cstandard(no,ni,x,ww,ju,isd,xs)
      if(isd .le. 0)goto 70499
70400 do 70401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
70401 continue
70450 continue
70499 continue
      call fgLTnet1(parm,no,ni,x,yq,yt,ycr,d,cr,g,ww,ju,vq,cl,ne,&
nx,nlam,flmin,ulam,thr,maxit,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 70999
70600 do 70601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
70601 continue
70699 continue
70999 continue
      deallocate(ww,ju,vq)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 71000~73999
      subroutine fgLTnet1(parm,no,ni,x,yq,yt,ycr,d,cr,g,q,ju,vp,cl,&
ne,nx,nlam,flmin,ulam,cthri,maxit,lmu,ao,m,kin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),ycr(no),q(no),g(no),vp(ni),ulam(nlam)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ju(ni),m(nx),kin(nlam),cr(no)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq,km!,kmq
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,jq,kq,mm,ixx
      integer, dimension (:), allocatable :: jcr,kcr,jpcr,kpcr,jqcr,kqcr
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
!      allocate(kmq(1:no),stat=ierr)
!      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jqcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kqcr(1:no),stat=ierr)
      jerr=jerr+ierr
! jq: orders of truncations times
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
! kq: mark the last index for each groups of unique event times in jq
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 71999
      call groupsFGLT(no,yq,yt,ycr,cr,q,nk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,t0,jerr)
      if(jerr.ne.0) go to 71999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 71011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 71021
71011 continue
      f=0.0d0
      e=q
71021 continue
71101 continue
      r0=riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
71200 do 71201 i=1,no
!      if(((yt(i) .ge. t0) .or. (cr(i).gt. 1) ).and. (q(i) .gt. 0.0d0))goto 71250
      w(i)=0.0d0
      wr(i)=w(i)
71250 continue
71201 continue
71299 continue
      call outerFGLT(no,nk,dq,dk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 71999
      if(flmin .ge. 1.0d0)goto 71301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
71301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
71400 do 71401 j=1,ni
      if(ju(j).eq.0)goto 71401
      ga(j)=abs(dot_product(wr,x(:,j)))
71401 continue
71499 continue
72000 do 72001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 72011
      al=ulam(ilm)
      goto 72099
72011 if(ilm .le. 2)goto 72021
      al=al*alf
      goto 72099
72021 if(ilm .ne. 1)goto 72031
      al=big
      goto 72098
72031 continue
      al0=0.0d0
72100 do 72101 j=1,ni
      if(ju(j).eq.0)goto 72101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
72101 continue
72199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
72098 continue
72099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
72200 do 72201 k=1,ni
      if(ixx(k).eq.1)goto 72201
      if(ju(k).eq.0)goto 72201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
72201 continue
72299 continue
73000 continue
73001 continue
73002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call vars(no,ni,x,w,ixx,v)

! Loop starts
73100 continue
73101 continue
      nlp=nlp+1
      dli=0.0d0
73200 do 73201 j=1,ni
      if(ixx(j).eq.0)goto 73201
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 73211
      at=0.0d0
      goto 73221
73211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
73221 continue
73231 continue
      if(at .eq. a(j))goto 73249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
      if(mm(j) .ne. 0)goto 73248
      nin=nin+1
      if(nin.gt.nx)goto 73299
      mm(j)=nin
      m(nin)=j
73248 continue
73249 continue
73201 continue
73299 continue
      if(nin.gt.nx)goto 73400
      if(dli.lt.cthr)goto 73400
      if(nlp .le. maxit)goto 73300
      jerr=-ilm
      return

73300 continue
73301 continue
73302 continue
      nlp=nlp+1
      dli=0.0d0
73310 do 73311 l=1,nin
      j=m(l)
      u=a(j)*v(j)+dot_product(wr,x(:,j))
      if(abs(u) .gt. vp(j)*sa)goto 73312
      at=0.0d0
      goto 73313
73312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
73313 continue
73314 continue
      if(at .eq. a(j))goto 73315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr=wr-del*w*x(:,j)
      f=f+del*x(:,j)
73315 continue
73311 continue
73319 continue
      if(dli.lt.cthr)goto 73331
      if(nlp .le. maxit)goto 73321
      jerr=-ilm
      return
73321 continue
      goto 73302
73331 continue
      goto 73101
73400 continue
      if(nin.gt.nx)goto 73499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outerFGLT(no,nk,dq,dk,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 73411
      jerr=jerr-ilm
      go to 71999
73411 continue
      ix=0
73420 do 73421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 73421
      ix=1
      goto 73429
73421 continue
73429 continue
      if(ix .ne. 0)goto 73498
73430 do 73431 k=1,ni
      if(ixx(k).eq.1)goto 73431
      if(ju(k).eq.0)goto 73431
      ga(k)=abs(dot_product(wr,x(:,k)))
      if(ga(k) .le. sa*vp(k))goto 73435
      ixx(k)=1
      ix=1
73435 continue
73431 continue
73439 continue
      if(ix.eq.1) go to 73000
      goto 73499
73498 continue
      goto 73002
73499 continue
      if(nin .le. nx)goto 73500
      jerr=-10000-ilm
      goto 72999
73500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)-r0)/rr
      if(ilm.lt.mnl)goto 72001
      if(flmin.ge.1.0d0)goto 72001
      me=0
73510 do 73511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
73511 continue
73999 continue
      if(me.gt.ne)goto 72999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 72999
      if(dev(ilm).gt.devmax)goto 72999
72001 continue
72999 continue
      g=f
71999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx,km,kcr,jcr,&
      kq,jq,kpcr,jpcr,kqcr,jqcr)
!      deallocate(kmq)
      return
      end


! get size for IPW at-risk information
! Label 74000~74999
      subroutine groupsFGLT(no,yq,yt,ycr,cr,q,nk,kp,jp,&
        kq,jq,kcr,jcr,kpcr,jpcr,kqcr,jqcr,km, &!kmq,
        t0,jerr)
      implicit double precision (a-h,o-z)
      double precision yq(no),yt(no),ycr(no),q(no),km(no)!,kmq(no)
      integer jp(no),jq(no),cr(no),kp(*),kq(*)
      integer jcr(no),jpcr(no),jqcr(no),kcr(*),kpcr(*),kqcr(*)
      integer, dimension (:), allocatable :: jpa,jyq,jycr

      allocate(jpa(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jyq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jycr(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 74999

74010 do 74011 j=1,no
      jpa(j)=j
      jyq(j)=j
      jycr(j)=j
74011 continue
74021 continue
! Get the order of yt, jpa
      call psort7(yt,jpa,1,no)

! Remove entries with non-positive weights from the order jpa
! nj is the number of positive weights
      nj=0
74030 do 74031 j=1,no
      if(q(jpa(j)).le.0.0d0)goto 74031
      nj=nj+1
      jpa(nj)=jpa(j)
74031 continue
74039 continue
      if(nj .ne. 0)goto 74099
! Error if no positive weights
      jerr=20000
      return
74099 continue

! sort the truncation/cr times of the remaining entries
      call psort7(yq(jpa),jyq,1,nj)
      jyq=jpa(jyq)
      call psort7(ycr(jpa),jycr,1,nj)
      jycr=jpa(jycr)

! Kaplan-Meier
      gj=1
      j=1
      jj=1
      jjj=1
      qar=0.0d0
74100 continue
74101 if(j.gt.nj)goto 74190
74102 if((cr(jpa(j)).gt. 0).or.(ycr(jpa(j)).gt.yt(jpa(j))))goto 74180
      yb=yt(jpa(j))
      qc=q(jpa(j))
      jb=j
74105 if(jjj.gt.nj)goto 74109
      if(ycr(jycr(jjj)).ge. yb)goto 74109
      if(cr(jycr(jjj)).gt.1) km(jycr(jjj))=gj
      if((yq(jycr(jjj)).lt.ycr(jycr(jjj))).and.&
        (yt(jycr(jjj)).gt.ycr(jycr(jjj)))) qar=qar-q(jycr(jjj))
      jjj=jjj+1
      goto 74105
74109 continue
74110 continue
74111 if(jb.eq.1)goto 74119
74112 if(yt(jpa(jb-1)).lt.yb)goto 74119
      jb=jb-1
      if(yt(jpa(jb)).le.ycr(jpa(jb))) qar=qar+q(jpa(jb))
      goto 74111
74119 continue
!      if(jb.gt.1) qar=qar-sum(q(jpa(1:(jb-1))))
74120 continue
74121 if(jj.gt.nj)goto 74129
      if(yq(jyq(jj)).ge.yb) goto 74129
!     kmq(jyq(jj))=gj
      if(yq(jyq(jj)).lt.ycr(jyq(jj)))qar=qar+q(jyq(jj))
      jj=jj+1
      goto 74121
74129 continue
!      qar = qar + sum(q(jyq(1:(jj-1))))
74130 continue
      j=j+1
74131 if(j.gt.nj)goto 74139
74132 if(yt(jpa(j)).gt.yb)goto 74139
74150 continue
74151 if((cr(jpa(j)).gt. 0).or.(ycr(jpa(j)).gt.yt(jpa(j)))) goto 74159
      qc=qc+q(jpa(j))
74159 continue
      goto 74130
74139 continue
74160 continue
      je=j-1
      gj=gj*(1-qc/qar)
74162 do 74169 i=jb,je
      if(cr(jpa(i)).le.1) km(jpa(i))=gj
      if(yt(jpa(i)).le.ycr(jpa(i))) qar=qar-q(jpa(i))
74169 continue
      goto 74101
74180 continue
      if(cr(jpa(j)).le.1) km(jpa(j))=gj
      if(yt(jpa(j)).le.ycr(jpa(j))) qar=qar-q(jpa(j))
      j=j+1
      goto 74101
74190 continue
74191 if(jjj.gt.nj)goto 74199
      if(cr(jycr(jjj)).gt.1) km(jycr(jjj))=gj
      jjj=jjj+1
      goto 74191
74199 continue

! j index for last event time in jpa
      j=nj
74200 continue
74201 if(cr(jpa(j)).eq. 1)goto 74209
      j=j-1
      goto 74201
74209 continue
74220 continue
! tk last event time
! jk index after tk
      tk=yt(jpa(j))
      jk=j+1
! Remove entries truncated after the last event time
      if(jk.gt.nj)goto 74299
      nnj=jk-1
74240 do 74241 j=jk,nj
      if(yq(jpa(j)).ge.tk)goto 74241
      nnj=nnj+1
      jpa(nnj)=jpa(j)
74241 continue
      nj=nnj
74299 continue


! Remove entries censored prior to the first event time
      j=1
      njp=1
      njpcr=0
74300 continue
74301 if(cr(jpa(j)).eq.1)goto 74330
      j=j+1
      goto 74301
74330 continue
      jp(1)=jpa(j)
      t0=yt(jpa(j))
      j0=j-1
74331 if(j0.eq.0)goto 74339
74332 if(yt(jpa(j0)).lt.t0)goto 74335
      if((cr(jpa(j0)).gt.1).and.(ycr(jpa(j0)).lt.t0))goto 74333
      njp=njp+1
      jp(njp)=jpa(j0)
      goto 74334
74333 continue
      njpcr=njpcr+1
      jpcr(njpcr)=jpa(j0)
74334 continue
      j0=j0-1
      goto 74331
74335 continue
      nj=nj-j0
      j=j-j0
74336 do 74338 i=1,nj
      jpa(i)=jpa(i+j0)
74338 continue
74339 continue
74399 continue

! sort the truncation/cr times of the remaining entries
      call psort7(yq(jpa),jyq,1,nj)
      jyq=jpa(jyq)
      call psort7(ycr(jpa),jycr,1,nj)
      jycr=jpa(jycr)


! nk number of unique event times
! kp mark the last index for each groups of unique event times in jp
! kq mark the last index for each groups of unique event times in jq
! kcr mark the last index for each groups of unique event times in jcr
! kpcr mark the last index for each groups of unique event times in jpcr
! kqcr mark the last index for each groups of unique event times in jqcr
! ip index in jpa
! iq index in jyq
! icr index in jycr
! njp index in jp
! njq index in jq
! njcr index in ijcr
! njpcr index in ijpcr
! njqcr index in ijqcr
74500 continue
      jerr=0
      nk=0
      yk=t0
      ip=j
      iq=1
      icr=1
      njq=0
      njcr=0
      njqcr=0
74510 continue
74511 if(iq.gt.nj)goto 74518
      i=jyq(iq)
74512 if(yq(i).ge.yk)goto 74518
74513 if(yt(i).lt.yk)goto 74517
74514 if((cr(i).gt.1).and.(ycr(i).lt.yq(i)))goto 74515
      njq=njq+1
      jq(njq)=i
      goto 74517
74515 continue
      njqcr=njqcr+1
      jqcr(njqcr)=i
74517 continue
      iq=iq+1
      goto 74510
74518 continue
      kq(nk+1)=njq
      kqcr(nk+1)=njqcr
74520 continue
74521 if(icr.gt.nj)goto 74528
      i=jycr(icr)
74522 if(ycr(i).ge.yk)goto 74528
      if((cr(i).le.1).or.(ycr(i).lt.yq(i)).or.(yt(i).lt.yk)) goto 74527
      njcr=njcr+1
      jcr(njcr)=i
74527 continue
      icr=icr+1
      goto 74520
74528 continue
      kcr(nk+1)=njcr

74600 continue
74601 ip=ip+1
74602 if(ip.gt.nj)goto 74699
      i = jpa(ip)
74611 if((yt(i).gt.yk) .and. (cr(i).eq.1))goto 74640
74612 if(yq(i).ge.yk)goto 74601
74613 if((cr(i).gt.1).and.(ycr(i).lt.yk))goto 74630
      njp=njp+1
      jp(njp)=i
      goto 74601
74630 continue
      njpcr=njpcr+1
      jpcr(njpcr)=i
      goto 74601
74640 continue
      njp=njp+1
      jp(njp)=i
      nk=nk+1
      kpcr(nk)=njpcr
      ykk=yk
      yk=yt(i)
! potential ties with yk to the front
      jj=njp-1
74641 if(yt(jp(jj)).lt.yk) goto 74648
      jj = jj - 1
      goto 74641
74648 continue
      kp(nk)=jj
      jp(njp)=jp(jj+1)
      jp(jj+1)=i
      jj=ip-1
74651 if(yt(jpa(jj)).lt.yk)goto 74658
      if(yq(jpa(jj)).lt.ykk)goto 74655
      njp=njp+1
      jp(njp)=jpa(jj)
74655 continue
      jj=jj-1
      goto 74651
74658 continue
      goto 74510
74699 continue
      nk=nk+1
      kp(nk)=njp
      kpcr(nk)=njpcr
74999 continue
      deallocate(jpa,jyq,jycr)
      return
      end

! log-likelihood
! Label: 75000~75999
      double precision function riskFGLT(no,nk,d,dk,f,e,kp,jp,kq,jq,&
        kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),f(no),km(no)
      integer kp(nk),jp(no),kq(nk),jq(no),kcr(nk),jcr(no)
      integer kpcr(nk),jpcr(no),kqcr(nk),jqcr(no)
      double precision e(no),u(nk)
      call uskFGLT(no,nk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,e,u)
      riskFGLT=dot_product(d,f)-dot_product(dk,log(u))
      return
      end

! total risks at event times
! Label: 76000~76999
      subroutine uskFGLT(no,nk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,e,u)
      implicit double precision (a-h,o-z)
      double precision e(no),u(nk),h,km(no)
      integer kp(nk),jp(no)
      integer kq(nk),jq(no),kcr(nk),jcr(no)
      integer kpcr(nk),jpcr(no),kqcr(nk),jqcr(no)

      h=0.0d0
      hG=0.0d0

! t_1 and t_2 are special cases
76100 continue
      g1 = km(jp(1))
76110 continue
      jj2=kq(1)
      jj1=1
      if(jj2 .lt. jj1) goto 76119
76111 do 76118 j=jj1,jj2
      h=h + e(jq(j))
76118 continue
76119 continue

76120 continue
      jj2=kqcr(1)
      jj1=1
      if(jj2 .lt. jj1) goto 76129
76121 do 76128 j=jj1,jj2
      i=jqcr(j)
      h=h + e(i)*g1/km(i)
      hG = hG + e(i)/km(i)
76128 continue
76129 continue

76130 continue
      jj2=kcr(1)
      jj1=1
      if(jj2 .lt. jj1) goto 76139
76131 do 76138 j=jj1,jj2
      i=jcr(j)
      h=h + e(i)*(g1/km(i)-1)
      hG = hG + e(i)/km(i)
76138 continue
76139 continue
      u(1)=h
76200 continue
      if(nk .lt. 2) goto 76299
76201 do 76290 k = 2,nk
      g0=g1
      g1=km(jp(kp(k-1)+1))
      h=h+hG*(g1-g0)
76210 continue
      jj2=kq(k)
      jj1=kq(k-1)+1
      if(jj2 .lt. jj1) goto 76219
76211 do 76218 j=jj1,jj2
      h=h + e(jq(j))
76218 continue
76219 continue

76220 continue
      jj2=kp(k-1)
      jj1=1
      if(k .gt. 2) jj1=kp(k-2)+1
      if(jj2 .lt. jj1) goto 76229
76221 do 76228 j=jj1,jj2
      h=h - e(jp(j))
76228 continue
76229 continue

76230 continue
      jj2=kqcr(k)
      jj1=kqcr(k-1)+1
      if(jj2 .lt. jj1) goto 76239
76231 do 76238 j=jj1,jj2
      i = jqcr(j)
      h=h + e(i)*g1/km(i)
      hG = hG + e(i)/km(i)
76238 continue
76239 continue

76240 continue
      jj2=kcr(k)
      jj1=kcr(k-1)+1
      if(jj2 .lt. jj1) goto 76249
76241 do 76248 j=jj1,jj2
      i = jcr(j)
      h=h - e(i)*(1-g1/km(i))
      hG = hG + e(i)/km(i)
76248 continue
76249 continue

76250 continue
      jj2=kpcr(k-1)
      jj1=1
      if(k.gt.2) jj1=kpcr(k-2)+1
      if(jj2 .lt. jj1) goto 76259
76251 do 76258 j=jj1,jj2
      i = jpcr(j)
      h=h - e(i)*g1/km(i)
      hG = hG - e(i)/km(i)
76258 continue
76259 continue
      u(k)=h
76290 continue
76299 continue
76999 continue
      return
      end

! two quantities for coordinate descent
! Label: 77000~77999
      subroutine outerFGLT(no,nk,d,dk,kp,jp,&
kq,jq,kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,e,wr,w,jerr,u)
      implicit double precision (a-h,o-z)
      double precision d(no),dk(nk),wr(no),w(no),km(no)
      double precision e(no),u(no),b,c
      integer kp(nk),jp(no),kcr(nk),jcr(no)
      integer kq(nk),jq(no),kqcr(nk),jqcr(no)
      integer kpcr(nk),jpcr(no)
      call uskFGLT(no,nk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,e,u)
      b=0.0d0
      bG=0.0d0
      c=0.0d0
      cG=0.0d0
      jerr=0
77000 continue
77001 do 77990 k=1,nk
77100 continue
      j1=1
      j2=kq(k)
      if(k.gt.1) j1=kq(k-1)+1
      if(j1.gt.j2)goto 77199
77101 do 77190 j=j1,j2
      i=jq(j)
      w(i)=-e(i)*(b-e(i)*c)
      wr(i)=d(i)+e(i)*b
77190 continue
77199 continue
77200 continue
      j1=1
      j2=kqcr(k)
      if(k.gt.1) j1=kqcr(k-1)+1
      if(j1.gt.j2)goto 77299
77201 do 77290 j=j1,j2
      i=jqcr(j)
      w(i)=-e(i)/km(i)*(bG-e(i)/km(i)*cG)
      wr(i)=e(i)/km(i)*bG
77290 continue
77299 continue
77300 continue
      j1=1
      j2=kcr(k)
      if(k.gt.1) j1=kcr(k-1)+1
77301 do 77390 j=j1,j2
      i=jcr(j)
      w(i)=w(i)+e(i)*(b-e(i)*c)-e(i)/km(i)*(bG-e(i)/km(i)*cG)
77311 continue
      wr(i)=wr(i)-e(i)*b+e(i)/km(i)*bG
77390 continue
77400 continue
      j1=1
      j2=kp(k)
      if(k.gt.1) j1=kp(k-1)+1
      gj=km(jp(j1))
      bk=dk(k)/u(k)
      ck=dk(k)/u(k)**2
      b=b+bk
      c=c+ck
      bG=bG+bk*gj
      cG=cG+ck*gj**2
77401 do 77490 j=j1,j2
      i=jp(j)
      w(i)=w(i)+e(i)*(b-e(i)*c)
      if(w(i) .ge. 0.0d0)goto 77411
      jerr=-30000
      return
77411 continue
      wr(i)=wr(i)-e(i)*b
77490 continue
77500 continue
      j1=1
      j2=kpcr(k)
      if(k.gt.1) j1=kpcr(k-1)+1
77501 do 77590 j=j1,j2
      i=jpcr(j)
      w(i)=w(i)+e(i)/km(i)*(bG-e(i)/km(i)*cG)
      if(w(i) .ge. 0.0d0)goto 77511
      jerr=-31000
      return
77511 continue
      wr(i)=wr(i)-e(i)/km(i)*bG
77590 continue
77990 continue
77999 continue
      return
      end

! log partial likelihood
! Label: 78000~78999
      subroutine fgLTloglik(no,ni,x,yq,yt,ycr,d,cr,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),yq(no),yt(no),ycr(no),g(no),w(no)
      double precision d(no),a(ni,nlam),flog(nlam)
      integer cr(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu,km
      integer, dimension (:), allocatable :: jp,kp,jq,kq,jcr,kcr
      integer, dimension (:), allocatable :: jpcr,kpcr,jqcr,kqcr
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jqcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kqcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 78999
78100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 78199
      jerr=9999
      go to 78999
78199 continue
78200 continue
      call groupsFGLT(no,yq,yt,ycr,cr,q,nk,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,t0,jerr)
      if(jerr.ne.0) goto 78999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
78300 do 78390 j=1,ni
      xm(j)=dot_product(q,x(:,j))/sw
78390 continue
78400 do 78490 lam=1,nlam
78410 do 78419 i=1,no
      f(i)=g(i)-gm+dot_product(a(:,lam),(x(i,:)-xm))
      e(i)=q(i)*exp(sign(min(abs(f(i)),fmax),f(i)))
78419 continue
      flog(lam)=riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)
78490 continue
78999 continue
      deallocate(e,uu,dk,f,jp,kp,jq,kq,dq,kcr,jcr,km,xm,q,jpcr,kpcr,jqcr,kqcr)
      end


! spfgnet and spfgLTnet: Label 80000~83999
! main double precision function: initialization and return
! Label 80000~80999
      subroutine spfgnet (parm,no,ni,x,ix,jx,yt,d,cr,g,w,jd,vp,cl,&
ne,nx,nlam,flmin,ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam),cr(no)
      double precision, dimension (:), allocatable :: xs,ww,vq,xm
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 80001
      jerr=10000
      return
80001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 80101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
80101 continue
      if(jerr.ne.0) return
      call spchkvars(no,ni,x,ix,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 80201
      jerr=7777
      return
80201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 80301
      jerr=9999
      return
80301 continue
      ww=ww/sw
      call spcstandard(no,ni,x,ix,jx,ww,ju,isd,xs,xm)
      if(isd .le. 0)goto 80499
80400 do 80401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
80401 continue
80450 continue
80499 continue
      call spfgnet1(parm,no,ni,x,ix,jx,xm,yt,d,cr,g,ww,&
ju,vq,cl,ne,nx,nlam,flmin,ulam,thr,maxit,lmu,ca,ia,&
nin,dev0,dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 80999
80600 do 80601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
80601 continue
80699 continue
80999 continue
      deallocate(ww,ju,vq,xm)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 81000~83999
      subroutine spfgnet1(parm,no,ni,x,ix,jx,xm,yt,d,cr,g,q,&
ju,vp,cl,ne,nx,nlam,flmin,ulam,cthri,maxit,lmu,ao,m,kin,dev0,&
dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),q(no),g(no),vp(ni),ulam(nlam),d(no)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),xm(ni)
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam),cr(no)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq,km
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,jcr,kcr,mm,ixx
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 81999
      call groupsFG(no,yt,cr,q,nk,kp,jp,kcr,jcr,km,t0,jerr)
      if(jerr.ne.0) go to 81999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 81011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 81021
81011 continue
      f=0.0d0
      e=q
81021 continue
81101 continue
      r0=riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
81200 do 81201 i=1,no
!      if(((yt(i) .ge. t0) .or. (cr(i).gt. 1) ).and. (q(i) .gt. 0.0d0))goto 81250
      w(i)=0.0d0
      wr(i)=w(i)
81250 continue
81201 continue
81299 continue
      call outerFG(no,nk,dq,dk,kp,jp,kcr,jcr,km,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 81999
      if(flmin .ge. 1.0d0)goto 81301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
81301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
81400 wrtot = sum(wr)
      do 81401 j=1,ni
      if(ju(j).eq.0)goto 81401
      jb=ix(j)
      je=ix(j+1)-1
      ga(j)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j))
81401 continue
81499 continue
82000 do 82001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 82011
      al=ulam(ilm)
      goto 82099
82011 if(ilm .le. 2)goto 82021
      al=al*alf
      goto 82099
82021 if(ilm .ne. 1)goto 82031
      al=big
      goto 82098
82031 continue
      al0=0.0d0
82100 do 82101 j=1,ni
      if(ju(j).eq.0)goto 82101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
82101 continue
82199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
82098 continue
82099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
82200 do 82201 k=1,ni
      if(ixx(k).eq.1)goto 82201
      if(ju(k).eq.0)goto 82201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
82201 continue
82299 continue
83000 continue
83001 continue
83002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call spvars(no,ni,x,ix,jx,xm,w,ixx,v)

! Loop starts
83100 continue
83101 continue
      nlp=nlp+1
      dli=0.0d0
83200 wrtot = sum(wr)
      do 83201 j=1,ni
      if(ixx(j).eq.0)goto 83201
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 83211
      at=0.0d0
      goto 83221
83211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
83221 continue
83231 continue
      if(at .eq. a(j))goto 83249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f = f - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
      if(mm(j) .ne. 0)goto 83248
      nin=nin+1
      if(nin.gt.nx)goto 83299
      mm(j)=nin
      m(nin)=j
83248 continue
83249 continue
83201 continue
83299 continue
      if(nin.gt.nx)goto 83400
      if(dli.lt.cthr)goto 83400
      if(nlp .le. maxit)goto 83300
      jerr=-ilm
      return

83300 continue
83301 continue
83302 continue
      nlp=nlp+1
      dli=0.0d0
83310 wrtot = sum(wr)
      do 83311 l=1,nin
      j=m(l)
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 83312
      at=0.0d0
      goto 83313
83312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
83313 continue
83314 continue
      if(at .eq. a(j))goto 83315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f = f - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
83315 continue
83311 continue
83319 continue
      if(dli.lt.cthr)goto 83331
      if(nlp .le. maxit)goto 83321
      jerr=-ilm
      return
83321 continue
      goto 83302
83331 continue
      goto 83101
83400 continue
      if(nin.gt.nx)goto 83499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outerFG(no,nk,dq,dk,kp,jp,kcr,jcr,km,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 83411
      jerr=jerr-ilm
      go to 81999
83411 continue
      indx=0
83420 do 83421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 83421
      indx=1
      goto 83429
83421 continue
83429 continue
      if(indx .ne. 0)goto 83498
83430 wrtot = sum(wr)
      do 83431 k=1,ni
      if(ixx(k).eq.1)goto 83431
      if(ju(k).eq.0)goto 83431
      jb=ix(k)
      je=ix(k+1)-1
      ga(k)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(k))
      if(ga(k) .le. sa*vp(k))goto 83435
      ixx(k)=1
      indx=1
83435 continue
83431 continue
83439 continue
      if(indx.eq.1) go to 83000
      goto 83499
83498 continue
      goto 83002
83499 continue
      if(nin .le. nx)goto 83500
      jerr=-10000-ilm
      goto 82999
83500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)-r0)/rr
      if(ilm.lt.mnl)goto 82001
      if(flmin.ge.1.0d0)goto 82001
      me=0
83510 do 83511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
83511 continue
83999 continue
      if(me.gt.ne)goto 82999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 82999
      if(dev(ilm).gt.devmax)goto 82999
82001 continue
82999 continue
      g=f
81999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx,km,kcr,jcr)
      return
      end


! spfgLTnet and auxiliaries: Label 84000~87999
! main double precision function: initialization and return
! Label 84000~84999
      subroutine spfgLTnet (parm,no,ni,x,ix,jx,yq,yt,ycr,d,cr,g,w,&
jd,vp,cl,ne,nx,nlam,flmin,ulam,thr,maxit,isd,lmu,ca,ia,nin,dev0,dev,alm,&
nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),ycr(no),g(no),w(no),vp(ni),ulam(nlam)
      double precision ca(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),jd(*),ia(nx),nin(nlam),cr(no)
      double precision, dimension (:), allocatable :: xs,xm,ww,vq
      integer, dimension (:), allocatable :: ju
      if(maxval(vp) .gt. 0.0d0)goto 84001
      jerr=10000
      return
84001 continue
      allocate(ww(1:no),stat=jerr)
      allocate(ju(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(vq(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(isd .le. 0)goto 84101
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
84101 continue
      if(jerr.ne.0) return
      call spchkvars(no,ni,x,ix,ju)
      if(jd(1).gt.0) ju(jd(2:(jd(1)+1)))=0
      if(maxval(ju) .gt. 0)goto 84201
      jerr=7777
      return
84201 continue
      vq=max(0.0d0,vp)
      vq=vq*ni/sum(vq)
      ww=max(0.0d0,w)
      sw=sum(ww)
      if(sw .gt. 0.0d0)goto 84301
      jerr=9999
      return
84301 continue
      ww=ww/sw
      call spcstandard(no,ni,x,ix,jx,ww,ju,isd,xs,xm)
      if(isd .le. 0)goto 84499
84400 do 84401 j=1,ni
      cl(:,j)=cl(:,j)*xs(j)
84401 continue
84450 continue
84499 continue
      call spfgLTnet1(parm,no,ni,x,ix,jx,xm,yq,yt,ycr,d,cr,g,&
ww,ju,vq,cl,ne,nx,nlam,flmin,ulam,thr,maxit,lmu,ca,ia,nin,dev0,&
dev,alm,nlp,jerr)
      if(jerr.gt.0) return
      dev0=2.0d0*sw*dev0
      if(isd .le. 0)goto 84999
84600 do 84601 k=1,lmu
      nk=nin(k)
      ca(1:nk,k)=ca(1:nk,k)/xs(ia(1:nk))
84601 continue
84699 continue
84999 continue
      deallocate(ww,ju,vq,xm)
      if(isd.gt.0) deallocate(xs)
      return
      end

! loop for coordinate descent
! Label 85000~87999
      subroutine spfgLTnet1(parm,no,ni,x,ix,jx,xm,yq,yt,ycr,d,cr,g,&
q,ju,vp,cl,ne,nx,nlam,flmin,ulam,cthri,maxit,lmu,ao,m,kin,dev0,&
dev,alm,nlp,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),ycr(no),q(no),g(no),vp(ni),ulam(nlam),xm(ni)
      double precision ao(nx,nlam),dev(nlam),alm(nlam),cl(2,ni),d(no)
      integer ix(*),jx(*),ju(ni),m(nx),kin(nlam),cr(no)
      double precision, dimension (:), allocatable :: w,dk,v,xs,wr,a,as,f,dq,km!,kmq
      double precision, dimension (:), allocatable :: e,uu,ga
      integer, dimension (:), allocatable :: jp,kp,jq,kq,mm,ixx
      integer, dimension (:), allocatable :: jcr,kcr,jpcr,kpcr,jqcr,kqcr
      call get_int_parms(sml,eps,big,mnlam,devmax,pmin,exmx)
      sml=sml*100.0d0
      devmax=devmax*0.99d0/0.999d0
! weighted relative risks
      allocate(e(1:no),stat=jerr)
! total relative risks at unique event times
! Note: log-bug in glmnent code.
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
! linear prediction
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(w(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
!      allocate(kmq(1:no),stat=ierr)
!      jerr=jerr+ierr
      allocate(v(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(a(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(as(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(xs(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ga(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(ixx(1:ni),stat=ierr)
      jerr=jerr+ierr
! Note: entries with non-positive weights, observation times before
! the first event time or truncation times after the last event time
! are excluded from the orders
!
! jp: orders of observation times excluding entries
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
! kp: mark the last index for each groups of unique event times in jp
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jqcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kqcr(1:no),stat=ierr)
      jerr=jerr+ierr
! jq: orders of truncations times
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
! kq: mark the last index for each groups of unique event times in jq
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
! dk: number/total weight of event ties at each unique event time
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(wr(1:no),stat=ierr)
      jerr=jerr+ierr
! dq: weighted event indicator
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(mm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 85999
      call groupsFGLT(no,yq,yt,ycr,cr,q,nk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,t0,jerr)
      if(jerr.ne.0) go to 85999
      alpha=parm
      oma=1.0d0-alpha
      nlm=0
      ixx=0
      al=0.0d0
      dq=d*q
! count the total weights/ties of event at each unique event time
      call died(no,nk,dq,kp,jp,dk)
      a=0.0d0
      f(1)=0.0d0
      fmax=log(huge(f(1))*0.1d0)
      if(nonzero(no,g) .eq. 0)goto 85011
      f=g-dot_product(q,g)
      e=q*exp(sign(min(abs(f),fmax),f))
      goto 85021
85011 continue
      f=0.0d0
      e=q
85021 continue
85101 continue
      r0=riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)
      rr=-(dot_product(dk(1:nk),log(dk(1:nk)))+r0)
      dev0=rr
85200 do 85201 i=1,no
!      if(((yt(i) .ge. t0) .or. (cr(i).gt. 1) ).and. (q(i) .gt. 0.0d0))goto 85250
      w(i)=0.0d0
      wr(i)=w(i)
85250 continue
85201 continue
85299 continue
      call outerFGLT(no,nk,dq,dk,kp,jp,kq,jq,kcr,jcr,&
      kpcr,jpcr,kqcr,jqcr,km,e,wr,w,jerr,uu)
      if(jerr.ne.0) go to 85999
      if(flmin .ge. 1.0d0)goto 85301
      eqs=max(eps,flmin)
      alf=eqs**(1.0d0/(nlam-1))
85301 continue
      m=0
      mm=0
      nlp=0
      nin=nlp
      mnl=min(mnlam,nlam)
      as=0.0d0
      cthr=cthri*dev0
85400 wrtot = sum(wr)
      do 85401 j=1,ni
      if(ju(j).eq.0)goto 85401
      jb=ix(j)
      je=ix(j+1)-1
      ga(j)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j))
85401 continue
85499 continue
86000 do 86001 ilm=1,nlam
      al0=al
      if(flmin .lt. 1.0d0)goto 86011
      al=ulam(ilm)
      goto 86099
86011 if(ilm .le. 2)goto 86021
      al=al*alf
      goto 86099
86021 if(ilm .ne. 1)goto 86031
      al=big
      goto 86098
86031 continue
      al0=0.0d0
86100 do 86101 j=1,ni
      if(ju(j).eq.0)goto 86101
      if(vp(j).gt.0.0d0) al0=max(al0,ga(j)/vp(j))
86101 continue
86199 continue
      al0=al0/max(parm,1.0d-3)
      al=alf*al0
86098 continue
86099 continue
      sa=alpha*al
      omal=oma*al
      tlam=alpha*(2.0d0*al-al0)
86200 do 86201 k=1,ni
      if(ixx(k).eq.1)goto 86201
      if(ju(k).eq.0)goto 86201
      if(ga(k).gt.tlam*vp(k)) ixx(k)=1
86201 continue
86299 continue
87000 continue
87001 continue
87002 continue
      if(nin.gt.0) as(m(1:nin))=a(m(1:nin))
      call spvars(no,ni,x,ix,jx,xm,w,ixx,v)

! Loop starts
87100 continue
87101 continue
      nlp=nlp+1
      dli=0.0d0
87200 wrtot = sum(wr)
      do 87201 j=1,ni
      if(ixx(j).eq.0)goto 87201
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 87211
      at=0.0d0
      goto 87221
87211 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
87221 continue
87231 continue
      if(at .eq. a(j))goto 87249
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
      if(mm(j) .ne. 0)goto 87248
      nin=nin+1
      if(nin.gt.nx)goto 87299
      mm(j)=nin
      m(nin)=j
87248 continue
87249 continue
87201 continue
87299 continue
      if(nin.gt.nx)goto 87400
      if(dli.lt.cthr)goto 87400
      if(nlp .le. maxit)goto 87300
      jerr=-ilm
      return

87300 continue
87301 continue
87302 continue
      nlp=nlp+1
      dli=0.0d0
87310 wrtot = sum(wr)
      do 87311 l=1,nin
      j=m(l)
      jb=ix(j)
      je=ix(j+1)-1
      u=a(j)*v(j)+dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(j)
      if(abs(u) .gt. vp(j)*sa)goto 87312
      at=0.0d0
      goto 87313
87312 continue
      at=max(cl(1,j),min(cl(2,j),sign(abs(u)-vp(j)*sa,u)/&
     (v(j)+vp(j)*omal)))
87313 continue
87314 continue
      if(at .eq. a(j))goto 87315
      del=at-a(j)
      a(j)=at
      dli=max(dli,v(j)*del**2)
      wr = wr + del*w*xm(j)
      wr(jx(jb:je))=wr(jx(jb:je))-del*w(jx(jb:je))*x(jb:je)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
87315 continue
87311 continue
87319 continue
      if(dli.lt.cthr)goto 87331
      if(nlp .le. maxit)goto 87321
      jerr=-ilm
      return
87321 continue
      goto 87302
87331 continue
      goto 87101
87400 continue
      if(nin.gt.nx)goto 87499
      e=q*exp(sign(min(abs(f),fmax),f))
      call outerFGLT(no,nk,dq,dk,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,e,wr,w,jerr,uu)
      if(jerr .eq. 0)goto 87411
      jerr=jerr-ilm
      go to 85999
87411 continue
      indx=0
87420 do 87421 j=1,nin
      k=m(j)
      if(v(k)*(a(k)-as(k))**2.lt.cthr)goto 87421
      indx=1
      goto 87429
87421 continue
87429 continue
      if(indx .ne. 0)goto 87498
87430 wrtot = sum(wr)
      do 87431 k=1,ni
      if(ixx(k).eq.1)goto 87431
      if(ju(k).eq.0)goto 87431
      jb=ix(k)
      je=ix(k+1)-1
      ga(k)=abs(dot_product(wr(jx(jb:je)),x(jb:je))-wrtot*xm(k))
      if(ga(k) .le. sa*vp(k))goto 87435
      ixx(k)=1
      indx=1
87435 continue
87431 continue
87439 continue
      if(indx.eq.1) go to 87000
      goto 87499
87498 continue
      goto 87002
87499 continue
      if(nin .le. nx)goto 87500
      jerr=-10000-ilm
      goto 86999
87500 continue
      if(nin.gt.0) ao(1:nin,ilm)=a(m(1:nin))
      kin(ilm)=nin
      alm(ilm)=al
      lmu=ilm
      dev(ilm)=(riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)-r0)/rr
      if(ilm.lt.mnl)goto 86001
      if(flmin.ge.1.0d0)goto 86001
      me=0
87510 do 87511 j=1,nin
      if(ao(j,ilm).ne.0.0d0) me=me+1
87511 continue
87999 continue
      if(me.gt.ne)goto 86999
      if((dev(ilm)-dev(ilm-mnl+1))/dev(ilm).lt.sml)goto 86999
      if(dev(ilm).gt.devmax)goto 86999
86001 continue
86999 continue
      g=f
85999 continue
      deallocate(e,uu,w,dk,v,xs,f,wr,a,as,jp,kp,dq,mm,ga,ixx,km,kcr,jcr,&
      kq,jq,kpcr,jpcr,kqcr,jqcr)
!      deallocate(kmq)
      return
      end

! log partial likelihood
! Label: 88000~88999
      subroutine spfgloglik(no,ni,x,ix,jx,yt,d,cr,g,w,nlam,a,flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),g(no),w(no),a(ni,nlam),flog(nlam),d(no)
      integer ix(*),jx(*),cr(no)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu,km
      integer, dimension (:), allocatable :: jp,kp,jcr,kcr
      flog(1)=1
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 88999
88100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 88199
      jerr=9999
      go to 88999
88199 continue
88200 continue
      call groupsFG(no,yt,cr,q,nk,kp,jp,kcr,jcr,km,t0,jerr)
      if(jerr.ne.0) goto 88999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
88300 do 88390 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
88390 continue
88400 continue
      f=g-gm
88410 do 88419 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 88418
      f  = f  - a(j,1)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j,1)*x(jb:je)
88418 continue
88419 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(1)=riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)
88440 if(nlam.lt.2) goto 88999
88441 do 88490 lam=2,nlam
88450 do 88459 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 88458
      del = a(j,lam)-a(j,lam-1)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
88458 continue
88459 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(lam)=riskFG(no,nk,dq,dk,f,e,kp,jp,kcr,jcr,km,uu)
88490 continue
88999 continue
      deallocate(e,uu,dk,f,jp,kp,dq,kcr,jcr,km,xm,q)
      end

! log partial likelihood
! Label: 89000~89999
      subroutine spfgLTloglik(no,ni,x,ix,jx,yq,yt,ycr,d,cr,g,w,nlam,a,&
        flog,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),ycr(no),d(no),g(no),w(no)
      double precision a(ni,nlam),flog(nlam)
      integer cr(no),ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,xm,dq,q
      double precision, dimension (:), allocatable :: e,uu,km
      integer, dimension (:), allocatable :: jp,kp,jq,kq,jcr,kcr
      integer, dimension (:), allocatable :: jpcr,kpcr,jqcr,kqcr
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(uu(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jqcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kqcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kpcr(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(km(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 89999
89100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 89199
      jerr=9999
      go to 89999
89199 continue
89200 continue
      call groupsFGLT(no,yq,yt,ycr,cr,q,nk,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,t0,jerr)
      if(jerr.ne.0) goto 89999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
89300 do 89390 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
89390 continue
89400 continue
      f=g-gm
89410 do 89419 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 89418
      f  = f  - a(j,1)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j,1)*x(jb:je)
89418 continue
89419 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(1)=riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)
89440 if(nlam.lt.2) goto 89999
89441 do 89490 lam=2,nlam
89450 do 89459 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 89458
      del = a(j,lam)-a(j,lam-1)
      f  = f  - del*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+del*x(jb:je)
89458 continue
89459 continue
      e = q*exp(sign(min(abs(f),fmax),f))
      flog(lam)=riskFGLT(no,nk,dq,dk,f,e,kp,jp,kq,jq,&
      kcr,jcr,kpcr,jpcr,kqcr,jqcr,km,uu)
89490 continue
89999 continue
      deallocate(e,uu,dk,f,jp,kp,jq,kq,dq,kcr,jcr,km,xm,q,jpcr,kpcr,jqcr,kqcr)
      end


! compute the U_i and S_i for nodewise LASSO
! Label: 90000~90999
      subroutine spnodewise(no,ni,nd,x,ix,jx,yt,d,g,w,a,ui,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yt(no),g(no),w(no),&
      a(ni),ui(nd,ni),d(no)
      integer ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,dq,q
      double precision, dimension (:), allocatable :: e,xm
      integer, dimension (:), allocatable :: jp,kp
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 90999
90100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 90199
      jerr=9999
      go to 90999
90199 continue
90200 continue
      call groups(no,yt,d,q,nk,kp,jp,t0,jerr)
      if(jerr.ne.0) goto 90999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
90300 do 90390 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
90390 continue
90400 continue
      f=g-gm
90410 do 90419 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 90419
      f  = f  - a(j)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j)*x(jb:je)
90419 continue
      e = q*exp(sign(min(abs(f),fmax),f))
90500 continue
      call spnodewiseU(no,ni,nd,nk,kp,jp,x,ix,jx,d,e,ui,jerr)
90999 continue
      deallocate(e,dk,f,jp,kp,dq,q,xm)
      end

! compute the U_i's for nodewise LASSO
! Label: 91000~91199
      subroutine spnodewiseU(no,ni,nd,nk,kp,jp,x,ix,jx,d,e,ui,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),e(no),ui(nd,ni),d(no)
      integer jp(no),kp(nk),ix(*),jx(*)
      double precision, dimension (:), allocatable :: h1,ek,xi
      integer, dimension (:), allocatable :: jdk
      allocate(h1(1:ni),stat=jerr)
      allocate(jdk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(ek(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xi(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)goto 91099
! weighted relative risks
91000 continue
      h0=0.0d0
      h1(:)=0.0d0
      idk=0
91001 do 91090 k=nk,1,-1
      ek(:)=0.0d0
      ndk=0
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
91020 do 91029 j=j2,j1,-1
      i=jp(j)
      ek(i)=e(i)
      if(d(i).le.0.0d0)goto 91029
      ndk=ndk+1
      jdk(ndk)=i
91029 continue
      h0=h0+sum(ek)
91030 do 91038 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 91038
      h1(j)=h1(j)+dot_product(ek(jx(jb:je)),x(jb:je))
91038 continue
91040 do 91049 j=1,ndk
      idk=idk+1
      if(idk.le.nd)goto 91041
      jerr=88888
      return
91041 continue
      i=jdk(j)
      ek(:)=0.0d0
      ek(i)=1.0d0
      xi(:)=0.0d0
91042 do 91048 jj=1,ni
      jb=ix(jj)
      je=ix(jj+1)-1
      if(je.lt.jb) goto 91048
      xi(jj)=dot_product(ek(jx(jb:je)),x(jb:je))
91048 continue
      ui(idk,:)=xi-h1/h0
91049 continue
91090 continue
91099 continue
      deallocate(h1,jdk,ek,xi)
      end

! compute the U_i and S_i for nodewise LASSO
! Label: 92000~92999
      subroutine spnodewiseLT(no,ni,nid,x,ix,jx,yq,yt,d,g,w,a,ui,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),yq(no),yt(no),g(no),w(no),&
      a(ni),ui(no,ni),d(no)
      integer ix(*),jx(*)
      double precision, dimension (:), allocatable :: dk,f,dq,q
      double precision, dimension (:), allocatable :: e,xm
      integer, dimension (:), allocatable :: jp,kp,jq,kq
! weighted relative risks
      allocate(e(1:no),stat=jerr)
      allocate(q(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(f(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kp(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(jq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(kq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dk(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(dq(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xm(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)go to 92999
92100 continue
      q=max(0.0d0,w)
      sw=sum(q)
      if(sw .gt. 0.0d0)goto 92199
      jerr=9999
      go to 92999
92199 continue
92200 continue
      call groupsLT(no,yq,yt,d,q,nk,kp,jp,kq,jq,t0,tk,jerr)
      if(jerr.ne.0) goto 92999
      fmax=log(huge(e(1))*0.1d0)
      dq=d*q
      call died(no,nk,dq,kp,jp,dk)
      gm=dot_product(q,g)/sw
92300 do 92390 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=0.0d0
      if(je.gt.jb) xm(j)=dot_product(q(jx(jb:je)),x(jb:je))/sw
92390 continue
92400 continue
      f=g-gm
92410 do 92419 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 92419
      f  = f  - a(j)*xm(j)
      f(jx(jb:je))=f(jx(jb:je))+a(j)*x(jb:je)
92419 continue
      e = q*exp(sign(min(abs(f),fmax),f))
92500 continue
      call spnodewiseULT(no,ni,nd,nk,kp,jp,kq,jq,x,ix,jx,d,e,ui,jerr)
92999 continue
      deallocate(e,dk,f,jp,kp,dq,q,jq,kq,xm)
      end

! compute the U_i's for nodewise LASSO
! Label: 93000~93199
      subroutine spnodewiseULT(no,ni,nd,nk,kp,jp,kq,jq,x,ix,jx,d,e,ui,jerr)
      implicit double precision (a-h,o-z)
      double precision x(*),e(no),ui(nd,ni),d(no)
      integer jp(no),kp(nk),jq(no),kq(nk),ix(*),jx(*)
      double precision, dimension (:), allocatable :: h1,ek,xi,ei
      allocate(h1(1:ni),stat=jerr)
      allocate(ek(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(ei(1:no),stat=ierr)
      jerr=jerr+ierr
      allocate(xi(1:ni),stat=ierr)
      jerr=jerr+ierr
      if(jerr.ne.0)goto 93099
! weighted relative risks
93000 continue
      h0=0.0d0
      h1(:)=0.0d0
      idk=0
      ek(:)=0d0
93001 do 93090 k=1,nk
      j2=kq(k)
      j1=1
      if(k.gt.1)j1=kq(k-1)+1
      if(j1.gt.j2)goto 93020
93010 do 93012 j=j1,j2
      i=jq(j)
      ek(i)=e(i)
93012 continue
      h0=h0+sum(ek)
93014 do 93018 j=1,ni
      jb=ix(j)
      je=ix(j+1)-1
      if(je.lt.jb) goto 93018
      h1(j)=h1(j)+dot_product(ek(jx(jb:je)),x(jb:je))
93018 continue
!      hk0=h0
!      hk1=h1
      ek(:)=0d0
      j2=kp(k)
      j1=1
      if(k.gt.1) j1=kp(k-1)+1
93020 do 93029 j=j1,j2
      i=jp(j)
      ek(i)=-e(i)
      if(d(i).le.0.0d0)goto 93029
      idk=idk+1
      if(idk.le.nd)goto 93021
      jerr=88888
      return
93021 continue
      ei(:)=0d0
      ei(i)=1d0
      xi(:)=0d0
93022 do 93028 jj=1,ni
      jb=ix(jj)
      je=ix(jj+1)-1
      if(je.lt.jb) goto 93028
      xi(jj)=dot_product(ek(jx(jb:je)),x(jb:je))
93028 continue
      ui(idk,:)=xi-h1/h0
93029 continue
93090 continue
93099 continue
      deallocate(h1,ek,xi,ei)
      end

! internal double precision functions
! get the order of an integer vector
! Label: 9000~9099
      subroutine psort7int (v,a,ii,jj)
      implicit integer (a-h,o-z)
!
!     puts into a the permutation vector which sorts v into
!     increasing order. the array v is not modified.
!     only elements from ii to jj are considered.
!     arrays iu(k) and il(k) permit sorting up to 2**(k+1)-1 elements
!
!     this is a modification of cacm algorithm #347 by r. c. singleton,
!     which is a modified hoare quicksort.
!
      dimension a(jj),v(jj),iu(20),il(20)
      integer t,tt
      integer a
      integer v
      m=1
      i=ii
      j=jj
9001 if (i.ge.j) go to 9008
9002 k=i
      ij=(j+i)/2
      t=a(ij)
      vt=v(t)
      if (v(a(i)).le.vt) go to 9003
      a(ij)=a(i)
      a(i)=t
      t=a(ij)
      vt=v(t)
9003 l=j
      if (v(a(j)).ge.vt) go to 9005
      a(ij)=a(j)
      a(j)=t
      t=a(ij)
      vt=v(t)
      if (v(a(i)).le.vt) go to 9005
      a(ij)=a(i)
      a(i)=t
      t=a(ij)
      vt=v(t)
      go to 9005
9004 a(l)=a(k)
      a(k)=tt
9005 l=l-1
      if (v(a(l)).gt.vt) go to 9005
      tt=a(l)
      vtt=v(tt)
9006 k=k+1
      if (v(a(k)).lt.vt) go to 9006
      if (k.le.l) go to 9004
      if (l-i.le.j-k) go to 9007
      il(m)=i
      iu(m)=l
      i=k
      m=m+1
      go to 9009
9007 il(m)=k
      iu(m)=j
      j=l
      m=m+1
      go to 9009
9008 m=m-1
      if (m.eq.0) return
      i=il(m)
      j=iu(m)
9009 if (j-i.gt.10) go to 9002
      if (i.eq.ii) go to 9001
      i=i-1
9010 i=i+1
      if (i.eq.j) go to 9008
      t=a(i+1)
      vt=v(t)
      if (v(a(i)).le.vt) go to 9010
      k=i
9011 a(k+1)=a(k)
      k=k-1
      if (vt.lt.v(a(k))) go to 9011
      a(k+1)=t
      go to 9010
      end

! get the order of a vector
! Label: 9100~9199
      subroutine psort7 (v,a,ii,jj)
      implicit double precision (a-h,o-z)
!
!     puts into a the permutation vector which sorts v into
!     increasing order. the array v is not modified.
!     only elements from ii to jj are considered.
!     arrays iu(k) and il(k) permit sorting up to 2**(k+1)-1 elements
!
!     this is a modification of cacm algorithm #347 by r. c. singleton,
!     which is a modified hoare quicksort.
!
      dimension a(jj),v(jj),iu(20),il(20)
      integer t,tt
      integer a
      double precision v
      m=1
      i=ii
      j=jj
9101 if (i.ge.j) go to 9108
9102 k=i
      ij=(j+i)/2
      t=a(ij)
      vt=v(t)
      if (v(a(i)).le.vt) go to 9103
      a(ij)=a(i)
      a(i)=t
      t=a(ij)
      vt=v(t)
9103 l=j
      if (v(a(j)).ge.vt) go to 9105
      a(ij)=a(j)
      a(j)=t
      t=a(ij)
      vt=v(t)
      if (v(a(i)).le.vt) go to 9105
      a(ij)=a(i)
      a(i)=t
      t=a(ij)
      vt=v(t)
      go to 9105
9104 a(l)=a(k)
      a(k)=tt
9105 l=l-1
      if (v(a(l)).gt.vt) go to 9105
      tt=a(l)
      vtt=v(tt)
9106 k=k+1
      if (v(a(k)).lt.vt) go to 9106
      if (k.le.l) go to 9104
      if (l-i.le.j-k) go to 9107
      il(m)=i
      iu(m)=l
      i=k
      m=m+1
      go to 9109
9107 il(m)=k
      iu(m)=j
      j=l
      m=m+1
      go to 9109
9108 m=m-1
      if (m.eq.0) return
      i=il(m)
      j=iu(m)
9109 if (j-i.gt.10) go to 9102
      if (i.eq.ii) go to 9101
      i=i-1
9110 i=i+1
      if (i.eq.j) go to 9108
      t=a(i+1)
      vt=v(t)
      if (v(a(i)).le.vt) go to 9110
      k=i
9111 a(k+1)=a(k)
      k=k-1
      if (vt.lt.v(a(k))) go to 9111
      a(k+1)=t
      go to 9110
      end

! computational parameters
! Label: 9200~9299
      subroutine get_int_parms(sml,eps,big,mnlam,rsqmax,pmin,exmx)
      implicit double precision (a-h,o-z)
      data sml0,eps0,big0,mnlam0,rsqmax0,pmin0,exmx0  /1.0d-5,1.0d-6,9.9d35,&
     5,0.999d0,1.0d-9,250.0d0/
      sml=sml0
      eps=eps0
      big=big0
      mnlam=mnlam0
      rsqmax=rsqmax0
      pmin=pmin0
      exmx=exmx0
      return
      entry chg_fract_dev(arg)
      sml0=arg
      return
      entry chg_dev_max(arg)
      rsqmax0=arg
      return
      entry chg_min_flmin(arg)
      eps0=arg
      return
      entry chg_big(arg)
      big0=arg
      return
      entry chg_min_lambdas(irg)
      mnlam0=irg
      return
      entry chg_min_null_prob(arg)
      pmin0=arg
      return
      entry chg_max_exp(arg)
      exmx0=arg
      return
      end

! check if any of the predictors is constant
! (for standardization)
! Label: 9300~9399
      subroutine chkvars(no,ni,x,ju)
      implicit double precision (a-h,o-z)
      double precision x(no,ni)
      integer ju(ni)
9300 do 9301 j=1,ni
      ju(j)=0
      t=x(1,j)
9310 do 9311 i=2,no
      if(x(i,j).eq.t)goto 9311
      ju(j)=1
      goto 9319
9311 continue
9319 continue
9301 continue
9399 continue
      return
      end

! standardize the predictors
! Label: 9400~9499
      subroutine cstandard (no,ni,x,w,ju,isd,xs)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),w(no),xs(ni)
      integer ju(ni)
9400 do 9401 j=1,ni
      if(ju(j).eq.0)goto 9401
      xm=dot_product(w,x(:,j))
      x(:,j)=x(:,j)-xm
      if(isd .le. 0)goto 9408
      xs(j)=sqrt(dot_product(w,x(:,j)**2))
      x(:,j)=x(:,j)/xs(j)
9408 continue
9401 continue
9499 continue
      return
      end


! compute the w'(x^2)
! Label: 9500~9599
      subroutine vars(no,ni,x,w,ixx,v)
      implicit double precision (a-h,o-z)
      double precision x(no,ni),w(no),v(ni)
      integer ixx(ni)
9500 do 9501 j=1,ni
      if(ixx(j).gt.0) v(j)=dot_product(w,x(:,j)**2)
9501 continue
9599 continue
      return
      end

! total weight of events at unique event times
! Label: 9600~9699
      subroutine died(no,nk,d,kp,jp,dk)
      double precision d(no),dk(nk)
      integer kp(nk),jp(no)
      dk(1)=sum(d(jp(1:kp(1))))
9600 do 9601 k=2,nk
      dk(k)=sum(d(jp((kp(k-1)+1):kp(k))))
9601 continue
9699 continue
      return
      end

! check if vector is all zero
! Label: 9700~9799
      integer function nonzero(n,v)
      double precision v(n)
      nonzero=0
9700 do 9701 i=1,n
      if(v(i) .eq. 0.0d0)goto 9711
      nonzero=1
      return
9711 continue
9701 continue
9799 continue
      return
      end

! check if any of the predictors is constant (sparse)
! (for standardization)
! Label: 8100~8199
      subroutine spchkvars(no,ni,x,ix,ju)
      implicit double precision (a-h,o-z)
      double precision x(*)
      integer ix(*),ju(ni)
8100 do 8101 j=1,ni
      ju(j)=0
      jb=ix(j)
      nj=ix(j+1)-jb
      if(nj.eq.0)goto 8101
      je=ix(j+1)-1
      if(nj .ge. no)goto 8141
8120 do 8121 i=jb,je
      if(x(i).eq.0.0d0)goto 8121
      ju(j)=1
      goto 8129
8121 continue
8129 continue
      goto 8131
8141 continue
      t=x(jb)
8160 do 8161 i=jb+1,je
      if(x(i).eq.t)goto 8161
      ju(j)=1
      goto 8169
8161 continue
8169 continue
8131 continue
8105 continue
8101 continue
8199 continue
      return
      end


! standardize the predictors (sparse)
! Need to save the means!
! Label: 8200~8299
      subroutine spcstandard (no,ni,x,ix,jx,w,ju,isd,xs,xm)
      double precision x(*),w(no),xs(ni),xm(ni)
      integer ix(*),jx(*),ju(ni)

8200 do 8201 j=1,ni
      if(ju(j).eq.0)goto 8201
      jb=ix(j)
      je=ix(j+1)-1
      xm(j)=dot_product(w(jx(jb:je)),x(jb:je))
!      x(jb:je)=x(jb:je)-xm(j)
      if(isd .le. 0)goto 8208
!      wzero = sum(w)-sum(w(jx(jb:je)))
!     xm(j)^2 = sum(w) * xm(j)^2
      xs(j)=sqrt(dot_product(w(jx(jb:je)),x(jb:je)**2)-xm(j)**2)
      x(jb:je)=x(jb:je)/xs(j)
      xm(j) = xm(j)/xs(j)
8208 continue
8201 continue
8299 continue
      return
      end

! compute the w'(x^2) (sparse)
! Label: 8300~8399
      subroutine spvars(no,ni,x,ix,jx,xm,w,ixx,v)
      implicit double precision (a-h,o-z)
      double precision x(*),w(no),v(ni), xm(ni)
      integer ix(*),jx(*),ixx(ni)
8300 do 8301 j=1,ni
      if(ixx(j).le.0) goto 8301
      jb=ix(j)
      je=ix(j+1)-1
      wzero = sum(w) - sum(w(jx(jb:je)))
      v(j)=dot_product(w(jx(jb:je)),(x(jb:je)-xm(j))**2)+wzero*xm(j)**2
8301 continue
8399 continue
      return
      end

