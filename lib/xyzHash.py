import numpy,pyopencl,pyopencl.array,clutil,time

class xyzHash(object):
	"""
	reads an xyzStripeHash from a file and holds the data in numpy arrays
	
	>>> h = xyzHash("../dat/pet.xyzHash")
	>>> print h
	5.0
	192
	6 7 6
	>>> h = xyzHash("../dat/lec.xyzHash")
	>>> print h
	9999.0
	150
	0 0 0
	"""
	def __init__(self, fname):
		super(xyzHash, self).__init__()
		self.fname = fname
		with open(fname) as o:
			self.grid_size = float(o.next())
			self.num_point = int(o.next())
			self.xsize,self.ysize,self.zsize = (int(x) for x in o.next().split())
			self.points = numpy.zeros(shape=(self.num_point,3),dtype=numpy.float32)
			for i in xrange(self.num_point):
				self.points[i] = [float(x) for x in o.next().split()]
			self.grid = numpy.zeros(shape=(self.zsize,self.ysize,self.xsize,2),dtype=numpy.int16)
			for iz in range(self.zsize):
				for iy in range(self.ysize):
					for ix in range(self.xsize):
						self.grid[iz,iy,ix] = [float(x) for x in o.next().split()]
			# for i in range(self.xsize*self.ysize*self.zsize):
			#	 self.grid.extend(float(x) for x in o.next().split())
			assert self.points.size==3*self.num_point
			assert self.grid.size  ==2*self.xsize*self.ysize*self.zsize
	
	def print_grid(self):
		print float(numpy.sum(self.grid[:,:,:,1]-self.grid[:,:,:,0])) / float(self.num_point)
		self.grid.shape = (self.grid.size/2,2)
		for i in xrange(self.grid.size/2):
			print self.grid[i]
	
	def __str__(self):
		s  = str(self.grid_size)+"\n"
		s += str(self.num_point)+"\n"
		s += str(self.xsize)+" "+str(self.ysize)+" "+str(self.zsize)
		return s
	
	def dump_pdb(self,fname="NONAME.pdb"):
		with open(fname,'w') as o:
			for i in xrange(self.points.shape[0]):
				x,y,z = self.points[i,:]
				o.write("HETATM %4i HASH XYZ A %3i     %7.3f %7.3f %7.3f  1.00  0.00\n"%(i+1,i/10+1,x,y,z))
	
	def get_coord_clarray(self,queue):
		if not hasattr(self,'cl_coord_array'):
			tmp = self.points.ravel()
			self.cl_coord_array = pyopencl.array.empty(queue,tmp.shape,dtype=numpy.float32)
			self.cl_coord_array.set(tmp)
		return self.cl_coord_array
	
	def get_grid_clarray(self,queue):
		if not hasattr(self,'cl_grid_array'):
			tmp = self.grid.ravel()
			if not len(tmp): return None
			self.cl_grid_array = pyopencl.array.empty(queue,tmp.shape,dtype=numpy.int16)
			self.cl_grid_array.set(tmp)
		return self.cl_grid_array
	

def test_xyzhash_cl():
	xh = xyzHash("../dat/hyd.xyzHash")
	with open("xyzHash.cl") as o: KERN = o.read()
	
	ctx,queue = clutil.my_get_gpu_context()

	nxyz,xsize,ysize,zsize,grid_size = xh.num_point,xh.xsize,xh.ysize,xh.zsize,xh.grid_size
	
	prg = pyopencl.Program(ctx,KERN%vars())
	prg.build("-I/Users/sheffler/project/protoprot/ligh2t/lib "+clutil.MATHOPTS)
	knl = prg.test_xyzhash
		
	NL,NG = 64,12
	out = pyopencl.array.empty(queue, (1,), dtype=numpy.int32); out.fill(0)
	clpts = xh.get_coord_clarray(queue)
	clgrid = xh.get_grid_clarray(queue)
	kargs = (clpts.data,clgrid.data,out.data)

	t = time.time()
	knl(queue, (NL,NG), (NL,1) , *kargs ).wait()
	print (time.time()-t)
	
	print 64*NL*NG
	print 28484
	print out

def test_xyzhash_cl2():
	xh = xyzHash("../dat/psi.xyzHash")
	with open("xyzHash.cl") as o: KERN = o.read()
	
	ctx,queue = clutil.my_get_gpu_context()

	nxyz,xsize,ysize,zsize,grid_size = xh.num_point,xh.xsize,xh.ysize,xh.zsize,xh.grid_size
	
	prg = pyopencl.Program(ctx,KERN%vars())
	prg.build("-I/Users/sheffler/project/protoprot/ligh2t/lib "+clutil.MATHOPTS)
	knl = prg.test_xyzhash
		
	NL,NG = 64,12
	out = pyopencl.array.empty(queue, (1,), dtype=numpy.int32); out.fill(0)
	clpts = xh.get_coord_clarray(queue)
	clgrid = xh.get_grid_clarray(queue)
	kargs = (clpts.data,clgrid.data,out.data)

	t = time.time()
	knl(queue, (NL,NG), (NL,1) , *kargs ).wait()
	print (time.time()-t)
	
	print 64*NL*NG
	print 53949
	print out


if __name__ == '__main__':
	# import doctest
	# tr = doctest.testmod()  
	# print "tests passed:",tr.attempted-tr.failed
	# print "tests failed:",tr.failed
	# for s in "pet hyd psi ldc ldn lec len".split():
	# 	xyzHash("../dat/%s.xyzHash"%s).dump_pdb("../test/%s_python.pdb"%s)
	test_xyzhash_cl()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
