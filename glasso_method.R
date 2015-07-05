library(glasso)
s=cov(t(covData),use = "pairwise")
g=glasso(s,rho=.01,maxit=1e2)