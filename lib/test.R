h = list()
for(i in c(2,5,8,10,20,30,40,50,60,70,80,90,100)) {
	x = read.table(paste("~/project/protoprot/ligh2t/test/hist_",i,sep=""))[[1]]	
	h[[as.character(i)]] = x
}

plot(0,type="n",xlim=c(0.0001,100),ylim=c(0.0001,max(sapply(h,max))))
for(n in names(h)) {
	lines(h[[n]]/sum(h[[n]]))
}



h = read.table("~/project/protoprot/ligh2t/test/hist6_50_50_1_12000000.dat")[[1]]

dim(h) = c(12,24,11,23,20,9)
image(log(apply(h,c(4,5),sum)),col=grey(99:0/100))
image((apply(h,c(4,5),sum)),col=grey(99:0/100))
image((apply(h,c(4,5),sum)),col=grey(999:0/1000))
image((apply(h,c(4,6),sum)),col=grey(999:0/1000))
image((apply(h,c(5,6),sum)),col=grey(999:0/1000))
image((apply(h,c(2,3),sum)),col=grey(999:0/1000))
image((apply(h,c(1,3),sum)),col=grey(999:0/1000))
