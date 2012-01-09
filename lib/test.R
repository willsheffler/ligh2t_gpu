h = list()
for(i in c(2,5,8,10,20,30,40,50,60,70,80,90,100)) {
	x = read.table(paste("~/project/protoprot/ligh2t/test/hist_",i,sep=""))[[1]]	
	h[[as.character(i)]] = x
}

plot(0,type="n",xlim=c(0.0001,100),ylim=c(0.0001,max(sapply(h,max))))
for(n in names(h)) {
	lines(h[[n]]/sum(h[[n]]))
}
