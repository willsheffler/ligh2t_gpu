h = array(0,c(11,300))
for(i in c(1,2,4,8,16,32,64,128,256,512,1024)) {
		tmp = read.table(paste("~/project/protoprot/ligh2t/test/hist_30_30_",i,"_1200000.dat",sep=""))[[1]]
		tmp = tmp * exp((1:300-0.5)/5)
		h[ip,ih,] = tmp / sum(tmp)
}




h = array(0,c(27,50,300))
for(ih in 3:50) {
	for(ip in c(16,21,27)) {
		tmp = read.table(paste("~/project/protoprot/ligh2t/test/hist_",i,sep=""))[[1]]
		tmp = tmp * exp((1:300-0.5)/5)
		h[ip,ih,] = tmp / sum(tmp)
	}
}














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

h = list()
for( i in c(10,20,30,40,50) ) {
for( j in c(10,20,30,40,50) ) {
	print(paste(paste(i,j)))
	h[[paste(i,j)]] = read.table(paste("~/project/protoprot/ligh2t/test/hist6_",i,"_",j,"_5_1200000.dat",sep=""))[[1]]
	dim(h[[paste(i,j)]]) = c(12,24,11,23,20,9)
}
}

par(mfrow=c(5,5),mar=c(0,0,0,0))
for( i in c(10,20,30,40,50) ) {
for( j in c(10,20,30,40,50) ) {
	image((apply(h[[paste(i,j)]],c(4,5),sum)),col=grey(96:0/97),breaks=0:97)
}
}

par(mfrow=c(1,1),mar=c(2,2,2,1))
plot(1,type='n',xlim=c(1,300),ylim=c(0.000001,0.1),log='y',main="temperature 1 2 4 8 16 32 64 128 256 512")
count = 1
hnl = list()
for(i in c(1,2,4,8,16,32,64,128,256,512)) {
	x = read.table(paste("~/project/protoprot/ligh2t/test/hist_0_0_",i,"_2400000.dat",sep=""))[[1]]
	lines(x/sum(x),col=heat.colors(10)[count],lwd=2)
	count = count+1
}
