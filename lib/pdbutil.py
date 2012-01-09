import gpu_mat_vec as gmv

def xform_pdb(xf,fnin,fnout):
	"""
	>>> from gpu_mat_vec import *
	>>> xf = xform( rotation_matrixf(1,0,0,3.14159), vec(10,0,0) )
	>>> xform_pdb(xf,"../pdb/hyda1_hash.pdb","../test/hyda1_xform.pdb")
	"""
	with  open(fnin)      as fin:
	 with open(fnout,'w') as out:
		for l in fin:
			if l.startswith("ATOM  ") or l.startswith("HETATM"):
				h,m,t = l[:31],l[31:54],l[54:]
				x,y,z = (float(z) for z in m.split())
				v = gmv.multxv(xf,gmv.vec(x,y,z))
				out.write(h+"%7.3f %7.3f %7.3f"%(v.x,v.y,v.z)+t)


def dump_pdb(ary,fname,c="A"):
	"""
	>>> import numpy,sys
	>>> dump_pdb( numpy.arange(12), sys.stdout )
	HETATM    1 HASH XYZ A   1       0.000   1.000   2.000  1.00  0.00
	HETATM    2 HASH XYZ A   1       3.000   4.000   5.000  1.00  0.00
	HETATM    3 HASH XYZ A   1       6.000   7.000   8.000  1.00  0.00
	HETATM    4 HASH XYZ A   1       9.000  10.000  11.000  1.00  0.00
	"""
	o = fname
	if type(fname) is type(""):
		o = open(fname,'w')
	for i in xrange(ary.size/3):
		x,y,z = ary[3*i+0:3*i+3]
		o.write("ATOM   %4i  CA  GPU %s %3i     %7.3f %7.3f %7.3f  1.00  0.00\n"%(i+1,c,i/10+1,x,y,z))
	if type(fname) is not type(o): o.close()

if __name__ == '__main__':
	import doctest
	tr = doctest.testmod()	
	print "tests passed:",tr.attempted-tr.failed
	print "tests failed:",tr.failed
