cv.phnet.deviance <-
  function (outlist, lambda, x, y, ftime,fstatus,weights, offset, foldid,
            grouped)
  {
    typenames = c(deviance = "Partial Likelihood Deviance")
    type.measure = "deviance"

    if (!is.null(offset)) {
      is.offset = TRUE
      offset = drop(offset)
    }
    else
    {
      is.offset = FALSE
      offset=rep(0,nrow(y))
    }
    is.fg=!is.null(fstatus)
    is.tdep=(ncol(y)==3)
    nfolds = max(foldid)
    if ((length(weights)/nfolds < 10) && !grouped) {
      warning("Option grouped=TRUE enforced for cv.coxnet, since < 3 observations per fold",
              call. = FALSE)
      grouped = TRUE
    }
    cvraw = matrix(NA, nfolds, length(lambda))
    ###We dont want to extrapolate lambdas on the small side
    mlami=max(sapply(outlist,function(obj)min(obj$lambda)))
    which_lam=lambda >= mlami

    for (i in seq(nfolds)) {
      i.fold = foldid == i
      fitobj = outlist[[i]]
      coefmat = predict(fitobj, type = "coeff",s=lambda[which_lam])
      if (grouped) {
        plfull = phnet.deviance(x = x, y = y, ftime=ftime,fstatus=fstatus,
                            offset = offset, weights = weights, beta = coefmat)
        fstatus.fold = ftime.fold=NULL
        if(is.fg)fstatus.fold=fstatus[!i.fold]
        if(is.tdep&&is.fg) ftime.fold=ftime[!i.fold]
        plminusk = phnet.deviance(x = x[!i.fold, ], y = y[!i.fold,],
                                  ftime=ftime.fold,fstatus=fstatus.fold,
                                  offset = offset[!i.fold], weights = weights[!i.fold],
                                   beta = coefmat)
        cvraw[i, seq(along = plfull)] = plfull - plminusk
      }
      else {
        fstatus.fold = ftime.fold=NULL
        if(is.fg)fstatus.fold=fstatus[!i.fold]
        if(is.tdep&&is.fg) ftime.fold=ftime[!i.fold]
        plk = phnet.deviance(x = x[i.fold, ], y = y[i.fold,],
                             ftime=ftime.fold,fstatus=fstatus.fold,
                             offset = offset[i.fold], weights = weights[i.fold],
                             beta = coefmat)
        cvraw[i, seq(along = plk)] = plk
      }
    }
    status = y[, "status"]
    N = nfolds - apply(is.na(cvraw), 2, sum)
    weights = as.vector(tapply(weights * status, foldid, sum))
    cvraw = cvraw/weights
    cvm = apply(cvraw, 2, weighted.mean, w = weights, na.rm = TRUE)
    cvsd = sqrt(apply(scale(cvraw, cvm, FALSE)^2, 2, weighted.mean,
                      w = weights, na.rm = TRUE)/(N - 1))
    out = list(cvm = cvm, cvsd = cvsd, name = typenames[type.measure])
    out
  }
