import sys,os,pyopencl,pyopencl,numpy,pyopencl.array,random,time,clint,operator,re,clutil
import matplotlib.pyplot as plt
import gpu_mat_vec as gmv
from xyzHash import xyzHash
from pdbutil import xform_pdb,dump_pdb
from gpu_mat_vec import nparray_to_XFORMs
from clint.textui import progress

OPTS = "-I/Users/sheffler/project/protoprot/ligh2t/lib -cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -w"

with open("testclash.cl") as f: KERN = f.read()%vars()
isubs = "npsi nhyd npet nlkp nlkh ntot nstat NGPUITER LS GS nxpsi nypsi nzpsi nxhyd nyhyd nzhyd nxpet nypet nzpet"
fsubs = "grid_size temperature"
ssubs = "hyd_stub pet_stub PSI_ET HYD_ET PET_ET"
for n in isubs.split(): KERN = re.sub(r"\b%s\b"%n,'%('+n+')i',KERN)
for n in fsubs.split(): KERN = re.sub(r"\b%s\b"%n,'%('+n+')ff',KERN)
for n in ssubs.split(): KERN = re.sub(r"\b%s\b"%n,'%('+n+')s',KERN)

def get_stub(components,name):
	import struct_data as sd
	if   components[3][-1]=="c":
		stub = ("stubrev(vec"+str(getattr(sd,name+'_N_N'))+
		               ",vec"+str(getattr(sd,name+'_N_CA'))+
		               ",vec"+str(getattr(sd,name+'_N_C'))+")" )
	elif components[3][-1]=="n":
		stub = ("stubrev(vec"+str(getattr(sd,name+'_C_C'))+
		               ",vec"+str(getattr(sd,name+'_C_CA'))+
		               ",vec"+str(getattr(sd,name+'_C_N'))+")" )
	else: assert "CAN'T DETERMINE ORIENTATIN OF HYDA1 LINKER!!!"==None
	return stub

def test():
	ctx,queue = clutil.my_get_gpu_context()
	LS,GS = 64,24
	NITER,NGPUITER = 20,5000
	lh,lp = 50,50
	nstat = 6
	for	temperature in (5,):
		# inputs
		components = "psi hyd pet lec ldn".split()
		hyd_stub = get_stub(components,"hyd")
		pet_stub = get_stub(components,"pet")
		PSI_ET = "vec(52.126627, 53.095875, 23.992003)"
		PET_ET = "vec( 5.468269, 20.041713, 11.962524)"
		HYD_ET = "vec(11.842096, 39.050866, 19.269006)"
		hashes = [xyzHash("../dat/%s.xyzHash"%n) for n in components]
		coords = [h.get_coord_clarray(queue) for h in hashes]
		grids  = [h.get_grid_clarray(queue) for h in hashes]
		nxpsi,nypsi,nzpsi = hashes[0].xsize, hashes[0].ysize, hashes[0].zsize
		nxhyd,nyhyd,nzhyd = hashes[1].xsize, hashes[1].ysize, hashes[1].zsize
		nxpet,nypet,nzpet = hashes[2].xsize, hashes[2].ysize, hashes[2].zsize
		grid_size = 5.0
		data = [d for d in reduce(operator.add,zip(coords,grids)) if d]
		npsi,nhyd,npet = (data[i].size/3 for i in (0,2,4))
		nlkh,nlkp = lh*3,lp*3
		ntot = nlkh+nlkp #npsi+nhyd+npet+nlkh+nlkp
		ranlux = pyopencl.array.empty(queue, (112*GS,)   , dtype=numpy.   int8); ranlux.fill(17)
		out    = pyopencl.array.empty(queue, (3*ntot*GS,), dtype=numpy.float32); out   .fill(0)
		xout   = pyopencl.array.empty(queue, (12*3*GS,)  , dtype=numpy.float32); xout  .fill(0)
		rout   = pyopencl.array.empty(queue, (GS,)       , dtype=numpy.float32); rout  .fill(0)
		vout   = pyopencl.array.empty(queue, (GS,6)      , dtype=numpy.float32); vout  .fill(0)
		status = pyopencl.array.empty(queue, (GS,2)      , dtype=numpy.  int32); status.fill(0)
		seed   = pyopencl.array.empty(queue, (1,)        , dtype=numpy. uint32); seed  .fill(0)
		stat   = pyopencl.array.empty(queue, (GS,nstat)  , dtype=numpy.  int32); stat  .fill(0)
		fstat  = pyopencl.array.empty(queue, (GS,nstat)  , dtype=numpy.float32); fstat .fill(0)
		hist   = pyopencl.array.empty(queue, (GS,300)    , dtype=numpy. uint32); hist  .fill(0)

		kargs = [x.data for x in data+[ranlux,seed,out,xout,rout,vout,status,stat,fstat,hist] ]
		my_test_kernel = pyopencl.Program(ctx,KERN%vars()).build(OPTS).my_test_kernel

		os.system("rm -f ../test/*.pdb")
		seed.fill(random.getrandbits(31))
		t = time.time()
		for ITER in range(1,NITER+1): 			
			my_test_kernel(queue,(LS,GS),(LS,1), *kargs ).wait();
			print ITER
		for i in range(GS):
			if stat.get()[i,1] == 0: continue
			if not ITER in vars(): ITER = 0
			ary = out.get()			
			fn = "../test/test_%i_%i_%i.pdb"%(i,lh,lp)
			dump_pdb(ary[i*len(ary)/GS:(i+1)*len(ary)/GS],"../test/.tmp")
			os.system("echo MODEL %i >> %s"%(ITER,fn))
			os.system("cat ../test/.tmp >> "+fn)			
			xform_pdb(nparray_to_XFORMs(xout.get())[3*i+0],"../pdb/hyda1_hash_bb.pdb","../test/.tmp")
			os.system("cat ../test/.tmp >> "+fn)
			xform_pdb(nparray_to_XFORMs(xout.get())[3*i+1],"../pdb/petf_hash_bb.pdb" ,"../test/.tmp")
			os.system("cat ../test/.tmp >> "+fn)
			# xform_pdb(nparray_to_XFORMs(xout.get())[3*i+2],"../pdb/petf_hash_bb.pdb" ,"../test/hyda_%i.pdb"%i)
			dump_pdb(vout.get()[i,],"../test/.tmp","X")
			os.system("cat ../test/.tmp >> "+fn)
			os.system("echo ENDMDL >> %s"%(fn))
		NOUT = NITER*NGPUITER*GS
		Nsamp = numpy.sum(stat.get()[:,0])
		print "COMPUTE RATE:",NOUT/(time.time()-t)
		h = numpy.apply_along_axis(numpy.sum, 0, hist.get())		
		numpy.savetxt("../test/hist_%i_%i_%f.dat"%(temperature,NOUT,sum(stat.get()[:,1]!=0)),h)
		print h[range(20)].reshape((2,10))
		print "samp frac: %f"%(float(Nsamp)/NOUT)
		# print
		print numpy.sum(stat.get()[:,1]!=0), "of",GS,"produced results"
		print numpy.transpose(status.get())
		print stat
		# plt.semilogy(h)
		# plt.show()

if __name__ == '__main__':
	test()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	