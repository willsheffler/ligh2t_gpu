import sys,os,pyopencl,pyopencl,numpy,pyopencl.array,random,time,clint,operator,re,clutil,glob
import matplotlib.pyplot as plt
import gpu_mat_vec as gmv
from xyzHash import xyzHash
from pdbutil import xform_pdb,dump_pdb
from gpu_mat_vec import nparray_to_XFORMs
from clint.textui import progress

OPTS = "-I. -cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -w"

#with open("nolinker.cl") as f: KERN = f.read()%vars()
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
	LS,GS = 64,int(sys.argv[2])
	NITER,NGPUITER = int(sys.argv[4]),int(sys.argv[3])
	lh,lp = 27,27
	NC = "n"
	temperature = 15.0
	nstat = 7
	# allocate
	ranlux = pyopencl.array.empty(queue, (112*GS,)   , dtype=numpy.   int8);      ranlux.fill(17)
	out    = pyopencl.array.empty(queue, (3*3*(100)*GS,), dtype=numpy.float32);    out   .fill(0) 
	xout   = pyopencl.array.empty(queue, (12*3*GS,)  , dtype=numpy.float32);      xout  .fill(0) 
	rout   = pyopencl.array.empty(queue, (GS,)       , dtype=numpy.float32);      rout  .fill(0) 
	vout   = pyopencl.array.empty(queue, (GS,6)      , dtype=numpy.float32);      vout  .fill(0) 
	status = pyopencl.array.empty(queue, (GS,2)      , dtype=numpy.  int32);      status.fill(0) 
	seed   = pyopencl.array.empty(queue, (1,)        , dtype=numpy. uint32);      seed  .fill(0) 
	stat   = pyopencl.array.empty(queue, (GS,nstat)  , dtype=numpy.  int32);      stat  .fill(0) 
	fstat  = pyopencl.array.empty(queue, (GS,nstat)  , dtype=numpy.float32);      fstat .fill(0) 
	hist   = pyopencl.array.empty(queue, (300,)      , dtype=numpy. uint32);      hist  .fill(0) 
	hist6  = pyopencl.array.empty(queue, (9,20,23,11,24,12), dtype=numpy.uint32); hist6.fill(0)  
	#for temperature in (5,10,9999999999):
	NC = 'c'
	for lp in (10,20,30,40,50,):
	  #for temperature in [2**x for x in range(6)]+[9e9]:
	    for lh in (10,20,30,40,50):
		NOUT = NITER*NGPUITER*GS
                #if len(glob.glob("../test/hist6_bind_%i_%i_%s_%i_%i_*.dat"%(lh,lp,NC,temperature,NOUT))) > 3: continue
	#for temperature in (15,):		
		#print temperature
		# inputs
		components = ("psi hyd pet ld%s len"%NC).split()
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
		out   .fill(0) 
		xout  .fill(0) 
		rout  .fill(0) 
		vout  .fill(0) 
		status.fill(0) 
		seed  .fill(0) 
		stat  .fill(0) 
		fstat .fill(0) 
		hist  .fill(0) 
		hist6.fill(0)  
		kargs = [x.data for x in data+[ranlux,seed,out,xout,rout,vout,status,stat,fstat,hist,hist6] ]
		my_test_kernel = pyopencl.Program(ctx,KERN%vars()).build(OPTS).my_test_kernel

		#os.system("rm -f ../test/*.pdb")
		seed.fill(random.getrandbits(31))
		t = time.time()
		for ITER in range(1,NITER+1):
			my_test_kernel(queue,(LS,GS),(LS,1), *kargs ).wait();			
			#print lh,lp,NC,ITER,"COMPUTE RATE:",ITER*NGPUITER*GS/(time.time()-t)/1000.0
		# for i in range(GS):
		# 	#if stat.get()[i,1] == 0: continue
		# 	if not ITER in vars(): ITER = 0
		# 	ary = out.get()			
		# 	fn = "../test/test_%i_%i_%i.pdb"%(i,lh,lp)
		# 	dump_pdb(ary[i*len(ary)/GS:(i+1)*len(ary)/GS],"../test/.tmp")
		# 	os.system("echo MODEL %i >> %s"%(ITER,fn))
		# 	os.system("cat ../test/.tmp >> "+fn)			
		# 	xform_pdb(nparray_to_XFORMs(xout.get())[3*i+0],"../pdb/hyda1_hash_bb.pdb","../test/.tmp")
		# 	os.system("cat ../test/.tmp >> "+fn)
		# 	xform_pdb(nparray_to_XFORMs(xout.get())[3*i+1],"../pdb/PetF_hash_bb.pdb" ,"../test/.tmp")
		# 	os.system("cat ../test/.tmp >> "+fn)
		# 	# xform_pdb(nparray_to_XFORMs(xout.get())[3*i+2],"../pdb/petf_hash_bb.pdb" ,"../test/hyda_%i.pdb"%i)
		# 	dump_pdb(vout.get()[i,],"../test/.tmp","X")
		# 	os.system("cat ../test/.tmp >> "+fn)
		# 	os.system("echo ENDMDL >> %s"%(fn))
		tag = str(random.random())
		Ntrial = numpy.sum(stat.get()[:,0])
		Nhist  = numpy.sum(stat.get()[:,2])
		h = hist.get()#numpy.apply_along_axis(numpy.sum, 0, hist.get())		
		print "DONE",lp,NC,lh,temperature,"mcmc steps/sec:",int(NOUT/(time.time()-t))
		#numpy.savetxt("../test/hist_%i_%i_%s_%i_%i_%s.dat"%(lh,lp,NC,temperature,NOUT,tag),h,"%i")
		#print h[range(20)].reshape((2,10))
		#print "samp frac: %f"%(float(Ntrial)/NOUT) , "sample loss:", float(Nhist-sum(h))/Nhist
		#print numpy.transpose(status.get())
		#print numpy.apply_along_axis(numpy.min,0,stat.get())
		#print numpy.apply_along_axis(numpy.max,0,stat.get())		
		#h6 = hist6.get()
		#numpy.savetxt("../test/hist6_%i_%i_%s_%i_%i_%s.dat"%(lh,lp,NC,temperature,NOUT,tag),h6.ravel(),"%i")

if __name__ == '__main__':
	test()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
