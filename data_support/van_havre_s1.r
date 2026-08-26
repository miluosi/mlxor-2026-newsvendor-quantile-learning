
# WARNING:  run on R version  3.1.0 and 3.1.1
# Also, some of these models are very time consuming to run; to obtain more rapid if less well supported results, one can
# reduce the number of iterations.


# install library of main functions
library(devtools) 
library(roxygen2)
install_github('zoevanhavre/Zmix') 
library(Zmix)


# Simulate HMMs: 

## PLOT SIMULATIONs ##########################
set.seed(222)

#pdf("SimHists.pdf", width=8, height=5)
	par(mfrow=c(2,2), mar=c( 5.1, 2.6 ,2.1, 1.5))
	hist(d1$Y, freq=F,main="Sim 1", xlab='Y', ylim=c(0, 0.28))
	lines(density(sim6func(n=100000)$Y), lty=2, lwd=2, col="firebrick1")
	
	hist(d2$Y, freq=F, main="Sim 2", xlab='Y', ylim=c(0,.25))
	lines(density(sim1func(n=100000)$Y),  lty=2, lwd=2, col="firebrick1")

	hist(d3$Y, freq=F,main="Sim 3", xlab='Y', ylim=c(0, 0.28))
	lines(density(sim2func(n=100000)$Y), lty=2, lwd=2, col="firebrick1")

	hist(d4$Y, freq=F,main="Sim 4",xlab='Y', ylim=c(0, 0.28))
	lines(density(sim5func(n=100000)$Y), lty=2, lwd=2, col="firebrick1")

#dev.off()


#
#	WARNING: 
#


#### PART 1: Exploratory simulations

set.seed(222)

# create datasets
S1n100<-sim6func(n=100) 
S2n100<-sim1func(n=100) 
S3n100<-sim2func(n=100)
S4n100<-sim5func(n=100)

S1n200<-sim6func(n=200) 
S2n200<-sim1func(n=200) 
S3n200<-sim2func(n=200)
S4n200<-sim5func(n=200)

# Run Zmix with tempering
S1n100.zmix<-Zmix_univ_tempered(y=S1n100, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S2n100.zmix<-Zmix_univ_tempered(y=S2n100, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S3n100.zmix<-Zmix_univ_tempered(y=S3n100, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S4n100.zmix<-Zmix_univ_tempered(y=S4n100, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S1n200.zmix<-Zmix_univ_tempered(y=S1n200, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S2n200.zmix<-Zmix_univ_tempered(y=S2n200, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S3n200.zmix<-Zmix_univ_tempered(y=S3n200, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))
S4n200.zmix<-Zmix_univ_tempered(y=S4n200, k=10,iter=50000,  tau=1, isSim=TRUE, alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6, 8, 10, 15, 20, 30))))

# Process results including label switching
## Plots will be created and saved in working directory
S1n100.zmix.pp<-Process_Output_Zmix(S1n100.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S1n100.zmix", SaveFileName="S1n100.zmix", PlotType="Boxplot")
S2n100.zmix.pp<-Process_Output_Zmix(S2n100.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S2n100.zmix", SaveFileName="S2n100.zmix", PlotType="Boxplot")
S3n100.zmix.pp<-Process_Output_Zmix(S3n100.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S3n100.zmix", SaveFileName="S3n100.zmix", PlotType="Boxplot")
S4n100.zmix.pp<-Process_Output_Zmix(S4n100.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S4n100.zmix", SaveFileName="S4n100.zmix", PlotType="Boxplot")
S1n200.zmix.pp<-Process_Output_Zmix(S1n200.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S1n200.zmix", SaveFileName="S1n200.zmix", PlotType="Boxplot")
S2n200.zmix.pp<-Process_Output_Zmix(S2n200.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S2n200.zmix", SaveFileName="S2n200.zmix", PlotType="Boxplot")
S3n200.zmix.pp<-Process_Output_Zmix(S3n200.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S3n200.zmix", SaveFileName="S3n200.zmix", PlotType="Boxplot")
S4n200.zmix.pp<-Process_Output_Zmix(S4n200.zmix, isSim=TRUE,  Burn=20000, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, makePlots=TRUE, Plot_Title="S4n200.zmix", SaveFileName="S4n200.zmix", PlotType="Boxplot")

# model statistics
S1n100.zmix.pp[[2]]
S2n100.zmix.pp[[2]]
S3n100.zmix.pp[[2]]
S4n100.zmix.pp[[2]]

S1n200.zmix.pp[[2]]
S2n200.zmix.pp[[2]]
S3n200.zmix.pp[[2]]
S4n200.zmix.pp[[2]]

# Estimated parameters
S1n100.zmix.pp[[1]]
S2n100.zmix.pp[[1]]
S3n100.zmix.pp[[1]]
S4n100.zmix.pp[[1]]

S1n200.zmix.pp[[1]]
S2n200.zmix.pp[[1]]
S3n200.zmix.pp[[1]]
S4n200.zmix.pp[[1]]





## Part 2: Replicate simulation
#
#
#
# requires new functions to run on a cluster in parallel. Note these will only work on Unix or Mac.
# Zmix with modified outputs to facilitate mering of parallel chains 
Zmix_lightLYRA<-function(y,  iter=20000){
				isSim=TRUE	
				k=10
				burn=iter/10
				alphas= c(30, 20, 10, 5, 3, 1, 0.5, 1/2^(c(2,3,4,5,6,7, 8, 10, 12, 15, 20, 30)))
			    ifelse(isSim==TRUE, Y<-y$Y, Y<-y)

				parallelAccept<-function(w1, w2, a1, a2){
					w1[w1< 1e-200]<-1e-200             # truncate so super small values dont crash everyting
					w2[w2< 1e-200]<-1e-200
					T1<-dDirichlet(w2, a1, log=TRUE)
					T2<-dDirichlet(w1, a2, log=TRUE)
					B1<-dDirichlet(w1, a1, log=TRUE)
					B2<-dDirichlet(w2, a2, log=TRUE)
					MH<-min(1,	exp(T1+T2-B1-B2)) 
					Ax<-sample(c(1,0), 1, prob=c(MH,1-MH))
					return(Ax)}
				nCh<-length(alphas)
				TrackParallelTemp<-matrix(nrow=iter, ncol=nCh)
				TrackParallelTemp[1,]<-c(1:nCh)
				tau=1
				n <-length(Y) 
				a=2.5;b=2/var(Y)
				d<-2
				lambda=sum(Y)/n  
		                                                                                                          # 1. set up priors
				mux<-list(mu=seq(from=min(Y), to=max(Y),length.out=k),sigma=rep(1, k),p=rep(1/k,k), k=k)
				n <-length(Y) 
				a=2.5; b<-0.5*var(Y);d<-2
				lambda<-sum(Y)/n  ;
					                                                                                                  # 2. set up matrices for parameters which will be saved
				map =    matrix(0,nrow = iter, ncol = 1)
				Loglike =   matrix(0,nrow = iter, ncol = 1)
				Bigmu = replicate(nCh,  matrix(0,nrow = iter, ncol = k)	, simplify=F)
				Bigsigma=replicate(nCh,  matrix(0,nrow = iter, ncol = k)	, simplify=F)
				Bigp =  replicate(nCh,  matrix(0,nrow = iter, ncol = k)	, simplify=F)
				Pzs =   replicate(nCh,  matrix(0,nrow = n, ncol = k)	, simplify=F)
				ZSaved=	replicate(nCh,  matrix(0,nrow = n, ncol = iter)	, simplify=F)
				K0Final<-matrix(nrow=iter, ncol=nCh)
					# start chains and  create inits needed for i=1 
					for (.ch in 1:nCh){
					Bigmu[[.ch]][1,] <- mux$mu                                                                        # initial value of mu's
					mu0=mux$mu		
					Bigp[[.ch]][1,] = mux$p                                                                           # inital value of p's
					p0=mux$p	
					Bigsigma[[.ch]][1,] = mux$sigma                                                                   # inits for sigma
					sig0=mux$sigma	}
								                                                                           #  		initialize chains:  ie. iteration 1:
					j<-1 
					for (.ch in 1:nCh){
					for (i in 1:n) {
					Pzs[[.ch]][i,]<-(p0/sqrt(sig0))*exp(-((Y[i]-mu0)^2)/(2*sig0))
			                                                                                                          
					Pzs[[.ch]][i,]<-Pzs[[.ch]][i,]/sum(Pzs[[.ch]][i,]) }}					
					
			                                                                                                          # 2 Make indicator matrix of assignments based on Pzs
			                                                                                                          #	sample 1 of the k classes for each row by Pzs (prob)
					for (.ch in 1:nCh){
					Z<-matrix()
					for (i in 1:n){Z[i]=sample((1:k),1, prob=Pzs[[.ch]][i,])}	
					matk = matrix((1:k), nrow = n, ncol = k, byrow = T)
					IndiZ = (Z == matk)		
					ZSaved[[.ch]][,1]<-Z
			                                                                                                          # 3 compute ns and sx
					ns = apply(IndiZ,2,sum)	
					ns[is.na(ns)]<-0  	
					sx = apply(IndiZ*Y, 2, sum)
					
			                                                                                                          # 4 Generate P[j](t) from dirichlet  (and save)
					Bigp[[.ch]][j,] = rdirichlet(m=1,par= ns+alphas[.ch])  
					
			                                                                                                          # 5	Generate Mu's   (and save)
					Bigmu[[.ch]][j,]<-rnorm(k,	mean=(lambda*tau+sx)/(tau+ns), sd=sqrt(Bigsigma[[.ch]][1,]/(tau+ns))) # must be sqrt as r takes in sd not var
					
			                                                                                                          #	ybar<-sx/ns
			                        Bigmu[[.ch]][j, is.na(Bigmu[[.ch]][j,])] <-0                                                                       # 6  Compute sv[j](t)	
					.bmu<- matrix((1:k), nrow = n, ncol = k, byrow = T)
					for (t in 1:n) {.bmu[t,]<-Bigmu[[.ch]][j,]}
					sv<-apply((Y*IndiZ-.bmu*IndiZ)^2, 2, sum)                                                         # changes, added /ns
			                                                                                                          # 7 Generate Sigma's (and save)
					Bigsigma[[.ch]][j,]<- rinvgamma(k, a+(ns+1)/2,	b+0.5*tau*(Bigmu[[.ch]][j,]-lambda)^2+0.5*sv)			                                                                                                          
					}					
			                                                                                                          # Log Likelihood: # Sum-n (log Sum-K ( weights x dnorm (y,thetas))) 
					for (i in 1:n){
					non0id<-c(1:k)[ns > 0]
					Loglike[j]<-Loglike[j]+ log(
						 sum( Bigp[[nCh]][j,non0id]*dnorm(Y[i], mean=Bigmu[[nCh]][j,non0id], sd=sqrt(Bigsigma[[nCh]][j,non0id]))))}
					
			                                                                                                          ### now finish loop for j>2
					for (j in 2:iter){
   					if( j  %% 200 == 0 ) cat(paste("iteration", j, "complete\n"))
					for (.ch in 1:nCh){
			                                                                                                    
					for (i in 1:n) {
						Pzs[[.ch]][i,]<-(Bigp[[.ch]][j-1,]/sqrt(Bigsigma[[.ch]][j-1,]))*exp(-((Y[i]-Bigmu[[.ch]][j-1,])^2)/(2*Bigsigma[[.ch]][j-1,]))    # 1. Generate P(Z[i](t)=j) 
						Pzs[[.ch]][i,]<-Pzs[[.ch]][i,]/sum(Pzs[[.ch]][i,])	# Scale to equal 1
						}
				                                                                                                          # 2 Make indicator matrix of assignments based on Pzs
			                                                                                                          #	sample 1 of the k classes for each row by Pzs (prob)
					for (i in 1:n){Z[i]=sample((1:k),1,replace=T, prob=Pzs[[.ch]][i,])}	
						matk = matrix((1:k), nrow = n, ncol = k, byrow = T)		
					IndiZ = (Z == matk)	  #indicator	
					ZSaved[[.ch]][,j]<-Z
			                                                                                                       
					ns = apply(IndiZ,2,sum)	   # 3 # compute ns and sx	
			                                                                                                       
					ns[is.na(ns)]<-0  # fix 'NA' number of obs to
					
					sx = apply(IndiZ*Y, 2, sum)
					
			                                                                                                          # 4 Generate P[j](t) from dirichlet  (and save)
					Bigp[[.ch]][j,] = rdirichlet(m=1,par= ns+alphas[.ch])
					
			                                                                                                          # 5	Generate Mu's   (and save)
					Bigmu[[.ch]][j,]<-rnorm(k,	mean=(lambda*tau+sx)/(tau+ns), sd=sqrt(Bigsigma[[.ch]][j-1,]/(tau+ns)))
					for (i in 1:length(Bigmu[[.ch]][j,])){ if ( is.na(Bigmu[[.ch]][j,i])) Bigmu[[.ch]][j,i]<-0 }
					
			                                                                                                          # 6 Compute sv[j](t)
					.bmu<- matrix((1:k), nrow = n, ncol = k, byrow = T)
					for (t in 1:n) {.bmu[t,]<-Bigmu[[.ch]][j,]}
					sv<-apply((Y*IndiZ-.bmu*IndiZ)^2, 2, sum)                                           		                                                                                                          
					Bigsigma[[.ch]][j,]<-  rinvgamma(k, a+(ns+1)/2,	b+0.5*tau*(Bigmu[[.ch]][j,]-lambda)^2+0.5*sv)
					
					# COUNT NON EMPTY
					K0Final[ j, .ch]<-sum(table(ZSaved[[.ch]][,j])>0)
					}
			           #new                                                                                               #### PARALLEL TEMPERING MOVES ###
					 if(j>1) {TrackParallelTemp[j,]<-TrackParallelTemp[j-1,]}      # SET para chains to previous values                                                                                           					
					if(j>20){
					if( sample(c(1,0),1, 0.33)==1){		 
					Chain1<-sample( 1:(nCh-1), 1)   
					Chain2<-Chain1+1
	                                                    # check ratio
					MHratio<- parallelAccept(Bigp[[Chain1]][j,], Bigp[[Chain2]][j,], rep(alphas[Chain1],k), rep(alphas[Chain2],k))
					if (MHratio==1){                                                                                  # switch the chains (weights, mean, sigma, Zs)
			             #new
			             .tpt1<-  TrackParallelTemp[j,Chain1 ]
			             .tpt2<-  TrackParallelTemp[j,Chain2 ]             
						TrackParallelTemp[j,Chain1 ]<-.tpt2
			            TrackParallelTemp[j,Chain2 ]<-.tpt1 
			                                                                                                   # Weights
					.p1<-	Bigp[[Chain1]][j,]
					.p2<-	Bigp[[Chain2]][j,]
					Bigp[[Chain1]][j,]<-.p2
					Bigp[[Chain2]][j,]<-.p1
			                                                                                                          # Means
					.m1<-	Bigmu[[Chain1]][j,]
					.m2<-	Bigmu[[Chain2]][j,]
					Bigmu[[Chain1]][j,]<-.m2
					Bigmu[[Chain2]][j,]<-.m1
			                                                                                                          # SD
					.s1<-	Bigsigma[[Chain1]][j,]
					.s2<-	Bigsigma[[Chain2]][j,]
					Bigsigma[[Chain1]][j,]<-.s2
					Bigsigma[[Chain2]][j,]<-.s1
					
			                                                                                                          # Zs
					.z1<-	ZSaved[[Chain1]][,j]
					.z2<-	ZSaved[[Chain2]][,j]
					ZSaved[[Chain1]][,j]<-.z2
					ZSaved[[Chain2]][,j]<-.z1
					}		}		}			
					}
					
					# trim out burn in
					colnames(K0Final)<-c(1:length(alphas))
					K0Final<-K0Final[ -1:-burn, ]
					K0Final<-melt(K0Final)
					K0Final[,3]<-factor(K0Final[,3], levels=c(1:k))
					table(K0Final[,2], K0Final[,3])/(iter-burn)

					}
# Function to run "NumRep" replicates of each simulation:
ReplicatedZmix_parallel<-function (NumRep,  sim , n, mylabels="Label", nIter=20000) {
	require('parallel')
 	# REPLICATE y samples
 	dtc<-min(detectCores(), NumRep)
 	print(paste ( "Using ",dtc, " Cores"))
	simMe<-function(whichOne=1, n){
		if(whichOne==1) return(sim6func(n))
		if(whichOne==2) return(sim1func(n))
		if(whichOne==3) return(sim2func(n))
		if(whichOne==4) return(sim5func(n))
		if(whichOne==5) return(sim2EASYfunc(n))}
	yrep<-lapply(rep(n, NumRep),  function(x) simMe( sim, x))
 	#zmixRun<-lapply(yrep, function(x){   Zmix_lightLYRA(x, K,...)} )
 	#zmixRun<-mclapply(yrep, FUN=function(x) Zmix_lightLYRA(x), mc.cores=dtc) 
 	zmixRun<-mclapply(yrep, FUN=function(x, it) Zmix_lightLYRA(x, it),it=nIter, mc.cores=dtc) 
 	
 	docall<-do.call(rbind, lapply(zmixRun, melt))
 	K0s<-data.frame(  "Replicate"=rep(1:NumRep, each= dim(docall)[1]/NumRep)  , docall)	
	names(K0s)[-1]<-c("PT_Chain", "K0", "Proportion")
	TargetK0<-subset(K0s, PT_Chain==max(K0s$PT_Chain))


	Ymatrix<-matrix(unlist(yrep), nrow=2*NumRep, byrow=TRUE)[seq(1,2*NumRep, by=2),]  # each row is a y
	Zmatrix<-matrix(unlist(yrep), nrow=2*NumRep, byrow=TRUE)[seq(2,2*NumRep, by=2),]
		# plots:
		p <- ggplot(TargetK0, aes(factor(K0), Proportion )) +  geom_boxplot()+xlab("Number of Groups")+ylab("Proportion of iterations")+geom_jitter(position=position_jitter(width=0.01,height=.01), alpha=.3, size=.5)+ coord_flip()
		ggsave(plot=p, filename= paste("Target_K0",mylabels,".tiff", sep="") ,
		 width=10, height=10, units='cm' )
	zout<-list("TargetK0"=TargetK0, "K0s"=K0s, "Y"=Ymatrix, "Z"=Zmatrix )
	save(zout, file =paste(mylabels, "Out.RDATA", sep="")) 
	}

Rep.Sim1n100<-ReplicatedZmix_parallel(NumRep=20,  sim=1 , n=100, mylabels="Rep.Sim1n100", nIter=20000)
Rep.Sim2n100<-ReplicatedZmix_parallel(NumRep=20,  sim=2 , n=100, mylabels="Rep.Sim2n100", nIter=20000)
Rep.Sim3n100<-ReplicatedZmix_parallel(NumRep=20,  sim=3 , n=100, mylabels="Rep.Sim3n100", nIter=20000)
Rep.Sim4n100<-ReplicatedZmix_parallel(NumRep=20,  sim=4 , n=100, mylabels="Rep.Sim4n100", nIter=20000)


Rep.Sim1n200<-ReplicatedZmix_parallel(NumRep=20,  sim=1 , n=100, mylabels="Rep.Sim1n200", nIter=20000)
Rep.Sim2n200<-ReplicatedZmix_parallel(NumRep=20,  sim=2 , n=100, mylabels="Rep.Sim2n200", nIter=20000)
Rep.Sim3n200<-ReplicatedZmix_parallel(NumRep=20,  sim=3 , n=100, mylabels="Rep.Sim3n200", nIter=20000)
Rep.Sim4n200<-ReplicatedZmix_parallel(NumRep=20,  sim=4 , n=100, mylabels="Rep.Sim4n200", nIter=20000)


# PArt 3 CASE STUDIES
#
#
#
#
set.seed(1)
data(Galaxy)
Galaxy.zmix <- Zmix_univ_tempered(Galaxy , tau=1, iter=50000, k=10) 
Galaxy.zmix.pp<-Process_Output_Zmix(runGalaxy2.zmix, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, isSim=FALSE, Plot_Title="Galaxy", SaveFileName="Galaxy_zmix", Burn=20000)
Galaxy.zmix.pp[[2]];Galaxy.zmix.pp[[1]]

data(Enzyme)
Enzyme.zmix <- Zmix_univ_tempered(Enzyme , tau=1, iter=50000, k=10) 
Enzyme.zmix.pp<-Process_Output_Zmix(runEnzyme2.zmix, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, isSim=FALSE, Plot_Title="Enzyme", SaveFileName="Enzyme_zmix", Burn=20000)
Enzyme.zmix.pp[[2]];Enzyme.zmix.pp[[1]]

data(Acidity)
Acidity.zmix <- Zmix_univ_tempered(Acidity , tau=1, iter=50000, k=10) 
Acidity.zmix.pp<-Process_Output_Zmix(runAcidity2.zmix, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, isSim=FALSE, Plot_Title="Acidity", SaveFileName="Acidity_zmix", Burn=20000)
Acidity.zmix.pp[[2]];Acidity.zmix.pp[[1]]


# secondary Run for Galaxy with smaller tau: 
set.seed(1)
Galaxy.zmix2 <- Zmix_univ_tempered(Galaxy , tau=0.01, iter=50000, k=10) 
Galaxy.zmix2.pp<-Process_Output_Zmix(runGalaxy2.zmix, LineUp=1, Pred_Reps=1000, Zswitch_Sensitivity=0.01, isSim=FALSE, Plot_Title="Galaxy, tau=0.01" , SaveFileName="Galaxy_zmix2", Burn=20000)
Galaxy.zmix2.pp[[2]];Galaxy.zmix2.pp[[1]]



