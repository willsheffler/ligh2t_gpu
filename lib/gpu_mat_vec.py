import math

## GPU-native functions
def mad(a,b,c):
	"""fused multiply-add as on GPU
	
	>>> mad(1,2,3)
	5.0
	"""
	return float(a)*float(b)+float(c)

def native_cos(x):
	"""Fast approx. cos on GPU. In python, just math.cos.
	
	>>> native_cos(math.pi)
	-1.0
	"""
	return math.cos(float(x))

def native_divide(x,y):
	"""Fast approx. division on GPU. In python, just x/y.
	
	>>> native_divide(5,4)
	1.25
	"""
	return float(x)/float(y)

def native_exp(x):
	"""Fast approx. exp on GPU. In python, just math.exp.
	
	>>> native_exp(1.0)
	2.718281828459045
	"""
	return math.exp(float(x))

def native_exp2(x):
	"""Fast approx. exp2 on GPU. In python, just math.pow(2,x).
	
	>>> native_exp2(1.0)
	2.0
	"""
	return math.pow(2,float(x))

def native_exp10(x):
	"""Fast approx. exp10 on GPU. In python, just math.pow(10,x).
	
	>>> native_exp10(1.0)
	10.0
	"""
	return math.pow(10,float(x))

def native_log(x):
	"""Fast approx. log on GPU. In python, just math.log.
	
	>>> native_log(4)
	1.3862943611198906
	"""
	return math.log(float(x),math.e)

def native_log2(x):
	"""Fast approx. log2 on GPU. In python, just math.log(x,2).
	
	>>> native_log2(4)
	2.0
	"""
	return math.log(float(x),2)

def native_log10(x):
	"""Fast approx. log10 on GPU. In python, just math.log(x,10).
	
	>>> native_log10(4)
	0.6020599913279623
	"""
	return math.log(float(x),10)

def native_powr(x,y):
	"""Fast approx. powr on GPU. In python, just math.pow.
	
	>>> native_powr(2,4)
	16.0
	"""
	return math.pow(float(x),float(y))

def native_recip(x):
	"""Fast approx. recip on GPU. In python, just 1/x.
	>>> native_recip(4)
	0.25
	"""
	return 1.0/float(x)

def native_rsqrt(x):
	"""Fast approx. rsqrt on GPU. In python, just 1/math.rsqrt.
	
	>>> native_rsqrt(4)
	0.5
	"""
	return 1.0/math.sqrt(float(x))

def native_sin(x):
	"""Fast approx. sin on GPU. In python, just math.sin.
	
	>>> native_sin(4)
	-0.7568024953079282
	"""
	return math.sin(float(x))

def native_sqrt(x):
	"""Fast approx. sqrt on GPU. In python, just math.sqrt.
	
	>>> native_sqrt(4)
	2.0
	"""
	return math.sqrt(float(x))

def native_tan(x):
	"""Fast approx. tan on GPU. In python, just math.tan.
	
	>>> native_tan(4)
	1.1578212823495775
	"""
	return math.tan(float(x))



class VEC:
	"""emulates my opencl 'struct VEC' in gpu_mat_vec.cl """
	def __init__(self):
		self.x = 0.0
		self.y = 0.0
		self.z = 0.0
	
	def __str__(self):
		"""
		>>> print VEC()
		VEC(0.000000,0.000000,0.000000)
		"""
		return "VEC(%f,%f,%f)"%(self.x,self.y,self.z)
	

def vec(x,y,z):
	"""
	>>> print vec(1,2,3)
	VEC(1.000000,2.000000,3.000000)
	"""
	v = VEC()
	v.x,v.y,v.z = float(x),float(y),float(z)
	return v


class MAT:
	"""emulates my opencl 'struct VEC' in gpu_mat_vec.cl """
	def __init__(self):
		"""
		>>> print MAT()
		MAT(0.000000,0.000000,0.000000 0.000000,0.000000,0.000000 0.000000,0.000000,0.000000 )
		"""
		self.xx = 0.0
		self.xy = 0.0
		self.xz = 0.0
		self.yx = 0.0
		self.yy = 0.0
		self.yz = 0.0
		self.zx = 0.0
		self.zy = 0.0
		self.zz = 0.0
	
	def __str__(self):
		return "MAT(%f,%f,%f %f,%f,%f %f,%f,%f )"%(self.xx,self.xy,self.xz,self.yx,self.yy,self.yz,self.zx,self.zy,self.zz)
	

def rowsf(xx, xy, xz, yx, yy, yz, zx, zy, zz):
	"""make MAT from floats in row-major order (??) oppsite of colsf
	
	>>> print rowsf(*range(9))
	MAT(0.000000,1.000000,2.000000 3.000000,4.000000,5.000000 6.000000,7.000000,8.000000 )
	"""
	m = MAT()
	m.xx=float(xx); m.xy=float(xy); m.xz=float(xz);
	m.yx=float(yx); m.yy=float(yy); m.yz=float(yz);
	m.zx=float(zx); m.zy=float(zy); m.zz=float(zz);
	return m

def colsf(xx, yx, zx, xy, yy, zy, xz, yz, zz):
	"""make MAT from floats in col-major order (??) oppsite of rowsf
	
	>>> print colsf(*range(9))
	MAT(0.000000,3.000000,6.000000 1.000000,4.000000,7.000000 2.000000,5.000000,8.000000 )
	"""
	m = MAT()
	m.xx=float(xx); m.xy=float(xy); m.xz=float(xz);
	m.yx=float(yx); m.yy=float(yy); m.yz=float(yz);
	m.zx=float(zx); m.zy=float(zy); m.zz=float(zz);
	return m;

def rows(rx, ry, rz):
	"""make MAT from floats in col-major order (??) oppsite of rowsf
	
	>>> print rows(*[vec(*range(3))]*3)
	MAT(0.000000,1.000000,2.000000 0.000000,1.000000,2.000000 0.000000,1.000000,2.000000 )
	"""
	assert isinstance(rx,VEC)
	assert isinstance(ry,VEC)
	assert isinstance(rz,VEC)
	m = MAT()
	m.xx=rx.x; m.xy=rx.y; m.xz=rx.z;
	m.yx=ry.x; m.yy=ry.y; m.yz=ry.z;
	m.zx=rz.x; m.zy=rz.y; m.zz=rz.z;
	return m;

def cols(cx, cy, cz):
	"""make MAT from floats in col-major order (??) oppsite of rowsf
	
	>>> print cols(*[vec(*range(3))]*3)
	MAT(0.000000,0.000000,0.000000 1.000000,1.000000,1.000000 2.000000,2.000000,2.000000 )
	"""
	assert isinstance(cx,VEC)
	assert isinstance(cy,VEC)
	assert isinstance(cz,VEC)
	m = MAT()
	m.xx=cx.x; m.xy=cy.x; m.xz=cz.x;
	m.yx=cx.y; m.yy=cy.y; m.yz=cz.y;
	m.zx=cx.z; m.zy=cy.z; m.zz=cz.z;
	return m;


class XFORM:
	"""emulates my opencl 'struct XFORM' in gpu_mat_vec.cl
	
	>>> print XFORM()                    #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(0.000000,0.000000,0.000000 0.000000,0.000000,0.000000 0.000000,0.000000,0.000000 )
		VEC(0.000000,0.000000,0.000000)
	"""
	def __init__(self):
		self.R = MAT()
		self.t = VEC()
	
	def __str__(self):
		return "XFORM:\n	"+str(self.R)+"\n	"+str(self.t)
	

def xform(R, t):
	"""make an XFORM from MAT and VEC
	
	>>> print xform( rowsf(*range(9)) , vec(9,10,11) )             #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(0.000000,1.000000,2.000000 3.000000,4.000000,5.000000 6.000000,7.000000,8.000000 )
		VEC(9.000000,10.000000,11.000000)
	"""
	assert isinstance(R,MAT)
	assert isinstance(t,VEC)
	x = XFORM()
	x.R = R
	x.t = t
	return x



def crossvv(a, b):
	"""cross VECs
	
	>>> print crossvv( vec(1,0,0), vec(0,1,0) )
	VEC(0.000000,0.000000,1.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	r = VEC();
	r.x = mad(a.y,b.z,-a.z*b.y);
	r.y = mad(a.z,b.x,-a.x*b.z);
	r.z = mad(a.x,b.y,-a.y*b.x);
	return r;

def addvv(a, b):
	"""add VECs
	
	>>> print addvv( vec(1,1,1), vec(1,2,3) )
	VEC(2.000000,3.000000,4.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	r = VEC();
	r.x = a.x+b.x;
	r.y = a.y+b.y;
	r.z = a.z+b.z;
	return r;

def subvv(a, b):
	"""subtract VECs
	
	>>> print subvv( vec(1,1,1), vec(1,2,3) )
	VEC(0.000000,-1.000000,-2.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	r = VEC();
	r.x = a.x-b.x;
	r.y = a.y-b.y;
	r.z = a.z-b.z;
	return r;

def dotvv(a, b):
	"""dot product of VECs
	
	>>> print dotvv( vec(1,1,0), vec(0,1,1) )
	1.0
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	return mad(a.x,b.x,mad(a.y,b.y,a.z*b.z));

def length2v(v):
	"""length squared of a VEC
	
	>>> print length2v( vec(1,1,1) )
	3.0
	"""
	assert isinstance(v,VEC)
	return mad(v.x,v.x,mad(v.y,v.y,v.z*v.z));

def length2f(x, y, z):
	"""length squared of an implicit VEC
	
	>>> print length2f( 1,1,1 )
	3.0
	"""
	return mad(float(x),float(x),mad(y,y,z*z));

def lengthv(v):
	"""length of a VEC
	
	>>> print lengthv( vec(1,1,1) )
	1.73205080757
	"""
	assert isinstance(v,VEC)
	return native_sqrt(mad(v.x,v.x,mad(v.y,v.y,v.z*v.z)));

def lengthf(x, y, z):
	"""length of an implicit VEC
	
	>>> print lengthf(1,1,1)
	1.73205080757
	"""
	return native_sqrt(mad(x,x,mad(y,y,z*z)));

def normalizedv(v):
	"""normalized VEC
	
	>>> print normalizedv( vec(1,2,3) )
	VEC(0.267261,0.534522,0.801784)
	"""
	assert isinstance(v,VEC)
	return multfv(native_recip(lengthv(v)) , v );

def normalizedf(x, y, z):
	"""normalized VEC from 3 floats
	
	>>> print normalizedf(1,2,3)
	VEC(0.267261,0.534522,0.801784)
	"""
	r = VEC();
	l = 1.0 / lengthf(x,y,z);
	r.x = x*l;
	r.y = y*l;
	r.z = z*l;
	return r;

def proj(a, v):
	"""project a VEC v onto another VEC a
	
	>>> print proj( vec(1,1,1), vec(1,2,3) )
	VEC(2.000000,2.000000,2.000000)
	"""
	assert isinstance(a,VEC)
	d = dotvv(a,v) / length2v(a);
	r = VEC();
	r.x = d*a.x;
	r.y = d*a.y;
	r.z = d*a.z;
	return r;

def pproj(a, v):
	"""project a VEC v perpendicular to another VEC a
	
	>>> p = pproj( vec(1,1,1), vec(1,2,3) )
	>>> print p
	VEC(-1.000000,0.000000,1.000000)
	>>> dotvv(p,vec(1,1,1))
	0.0
	"""
	assert isinstance(a,VEC)
	d = native_divide(dotvv(a,v), length2v(a) );
	r = VEC();
	r.x = v.x-d*a.x;
	r.y = v.y-d*a.y;
	r.z = v.z-d*a.z;
	return r;



def projection_matrix(a):
	"""get projection matrix onto VEC a
	
	>>> print projection_matrix( vec(1,2,3) )
	MAT(0.071429,0.142857,0.214286 0.142857,0.285714,0.428571 0.214286,0.428571,0.642857 )
	"""
	assert isinstance(a,VEC)
	P = MAT();
	l2 = 1.0/length2v(a);
	P.xx=l2*a.x*a.x; P.xy=l2*a.x*a.y; P.xz=l2*a.x*a.z;
	P.yx=l2*a.y*a.x; P.yy=l2*a.y*a.y; P.yz=l2*a.y*a.z;
	P.zx=l2*a.z*a.x; P.zy=l2*a.z*a.y; P.zz=l2*a.z*a.z;
	return P;

def projection_matrixf(x, y, z):
	"""get projection matrix onto implicet VEC a
	
	>>> print projection_matrixf(1,2,3)
	MAT(0.071429,0.142857,0.214286 0.142857,0.285714,0.428571 0.214286,0.428571,0.642857 )
	"""
	P = MAT();
	l2 = native_divide(1.0,length2f(x,y,z));
	P.xx=l2*x*x; P.xy=l2*x*y; P.xz=l2*x*z;
	P.yx=l2*y*x; P.yy=l2*y*y; P.yz=l2*y*z;
	P.zx=l2*z*x; P.zy=l2*z*y; P.zz=l2*z*z;
	return P;

def rotation_matrix(a, t):
	"""get rotation matrix, t in radians
	
	>>> print rotation_matrix( vec(1,0,0), math.pi )
	MAT(1.000000,0.000000,0.000000 0.000000,-1.000000,-0.000000 0.000000,0.000000,-1.000000 )
	"""
	assert isinstance(a,VEC)
	n = normalizedv(a);
	sin_t = native_sin(t);
	cos_t = native_cos(t);
	R = multfm(1.0-cos_t, projection_matrix(n));
	R.xx += cos_t;       R.xy -= sin_t * n.z; R.xz += sin_t * n.y;
	R.yx += sin_t * n.z; R.yy += cos_t;       R.yz -= sin_t * n.x;
	R.zx -= sin_t * n.y; R.zy += sin_t * n.x; R.zz += cos_t;
	return R;

def rotation_matrixf(x, y, z, t):
	"""get rotation matrix, t in radians
	
	>>> print rotation_matrixf(1, 0, 0, math.pi )
	MAT(1.000000,0.000000,0.000000 0.000000,-1.000000,-0.000000 0.000000,0.000000,-1.000000 )
	"""
	n = normalizedf(x,y,z);
	sin_t = native_sin(t );
	cos_t = native_cos(t );
	R = multfm(1.0-cos_t, projection_matrix(n));
	R.xx += cos_t;       R.xy -= sin_t * n.z; R.xz += sin_t * n.y;
	R.yx += sin_t * n.z; R.yy += cos_t;       R.yz -= sin_t * n.x;
	R.zx -= sin_t * n.y; R.zy += sin_t * n.x; R.zz += cos_t;
	return R;

def transposed(m):
	"""MAT transpose
	
	>>> print transposed( rotation_matrixf(1,0,0,1) )
	MAT(1.000000,0.000000,0.000000 0.000000,0.540302,0.841471 0.000000,-0.841471,0.540302 )
	"""
	assert isinstance(m,MAT)
	return colsf(m.xx,m.xy,m.xz,m.yx,m.yy,m.yz,m.zx,m.zy,m.zz);



def multmm(a, b):
	"""multiply two MATs
	
	>>> print multmm( rotation_matrixf(1,0,0,1), rotation_matrixf(1,0,0,-1) )
	MAT(1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,1.000000 )
	"""
	assert isinstance(a,MAT)
	assert isinstance(b,MAT)
	c = MAT()
	c.xx = mad(a.xx,b.xx,mad(a.xy,b.yx,a.xz*b.zx));
	c.xy = mad(a.xx,b.xy,mad(a.xy,b.yy,a.xz*b.zy));
	c.xz = mad(a.xx,b.xz,mad(a.xy,b.yz,a.xz*b.zz));
	c.yx = mad(a.yx,b.xx,mad(a.yy,b.yx,a.yz*b.zx));
	c.yy = mad(a.yx,b.xy,mad(a.yy,b.yy,a.yz*b.zy));
	c.yz = mad(a.yx,b.xz,mad(a.yy,b.yz,a.yz*b.zz));
	c.zx = mad(a.zx,b.xx,mad(a.zy,b.yx,a.zz*b.zx));
	c.zy = mad(a.zx,b.xy,mad(a.zy,b.yy,a.zz*b.zy));
	c.zz = mad(a.zx,b.xz,mad(a.zy,b.yz,a.zz*b.zz));
	return c;

def multmv(a, b):
	"""multiply MAT times VEC
	
	>>> print multmv( rotation_matrixf(1,0,0,math.pi/4.0), vec(0,1,0) )
	VEC(0.000000,0.707107,0.707107)
	"""
	assert isinstance(a,MAT)
	assert isinstance(b,VEC)
	c = VEC()
	c.x = mad(a.xx,b.x,mad(a.xy,b.y,a.xz*b.z));
	c.y = mad(a.yx,b.x,mad(a.yy,b.y,a.yz*b.z));
	c.z = mad(a.zx,b.x,mad(a.zy,b.y,a.zz*b.z));
	return c;

def multfv(a, v):
	"""multiply scalar times VEC element-wise
	
	>>> print multfv( 2.5, vec(1,1,0) )
	VEC(2.500000,2.500000,0.000000)
	"""
	assert isinstance(v,VEC)
	r = VEC()
	r.x = float(a)*v.x;
	r.y = float(a)*v.y;
	r.z = float(a)*v.z;
	return r;

def multfm(a, m):
	"""multiply scalar times MAT element-wise
	
	>>> print multfv( 2.5, vec(1,1,0) )
	VEC(2.500000,2.500000,0.000000)
	"""
	assert isinstance(m,MAT)
	r = MAT();
	r.xx= float(a)*m.xx;  r.xy= float(a)*m.xy;  r.xz= float(a)*m.xz;
	r.yx= float(a)*m.yx;  r.yy= float(a)*m.yy;  r.yz= float(a)*m.yz;
	r.zx= float(a)*m.zx;  r.zy= float(a)*m.zy;  r.zz= float(a)*m.zz;
	return r;

def multxx(x2, x1):
	"""apply one XFORM x2 to another x1
	
	>>> x1 = xform( rotation_matrixf(1,0,0,1), vec(10,0,0))
	>>> x2 = xform( rotation_matrixf(1,0,0,-1), vec(-10,0,0))
	>>> print multxx(x1,x2)               #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,1.000000 )
		VEC(0.000000,0.000000,0.000000)	
	"""
	assert isinstance(x1,XFORM)
	assert isinstance(x2,XFORM)
	x = XFORM();
	x.R = multmm(x2.R,x1.R);
	x.t = addvv(multmv(x2.R,x1.t),x2.t);
	return x;

def multxv(x, v):
	"""apply XFORM to a VEC
	
	>>> x1 = xform( rotation_matrixf(1,0,0,1), vec(10,0,0))
	>>> print multxv(x1, vec(1,0,0) )
	VEC(11.000000,0.000000,0.000000)
	>>> print multxv(x1,vec(0,1,0))
	VEC(10.000000,0.540302,0.841471)
	"""
	assert isinstance(x,XFORM)
	assert isinstance(v,VEC)	
	return addvv(multmv(x.R,v),x.t);



def xrev(x):
	"""inverse of XFORM
	
	>>> x1 = xform( rotation_matrixf(1,0,0,1), vec(1,0,0) )
	>>> print multxx( xrev(x1), x1 )                               #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,1.000000 )
		VEC(0.000000,0.000000,0.000000)
	"""
	assert isinstance(x,XFORM)
	r = XFORM();
	r.R = transposed(x.R);
	r.t = multmv(r.R,multfv(-1.0,x.t));
	return r;

def vvcxform(_x1, _x2, _y1, _y2,  _c1, _c2):
	"""get XFORM that takes one coord frame to another
	
	>>> x = vvcxform( vec(1,0,0), vec(0,1,0), vec(0,1,0), vec(1,0,0), vec(0,0,0), vec(10,10,10) )
	>>> print x                       #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(0.000000,1.000000,0.000000 1.000000,0.000000,0.000000 0.000000,0.000000,-1.000000 )
		VEC(10.000000,10.000000,10.000000)
	>>> print rotation_matrixf(1,1,0,math.pi)
	MAT(0.000000,1.000000,0.000000 1.000000,0.000000,-0.000000 -0.000000,0.000000,-1.000000 )
	"""
	assert isinstance(_x1,VEC)
	assert isinstance(_x2,VEC)
	assert isinstance(_y1,VEC)
	assert isinstance(_y2,VEC)
	assert isinstance(_c1,VEC)
	assert isinstance(_c2,VEC)		
	x1 = normalizedv(_x1);
	y1 = normalizedv(pproj(_x1,_y1));
	z1 = crossvv(x1,y1);
	x2 = normalizedv(_x2);
	y2 = normalizedv(pproj(_x2,_y2));
	z2 = crossvv(x2,y2);
	Rto = cols(x2,y2,z2);
	Rfr = rows(x1,y1,z1);
	x = XFORM();
	x.R = multmm(Rto,Rfr);
	x.t = subvv(_c2,multmv(x.R,_c1));
	return x;

def stub(a, b, c):
	"""get an XFORM which is a rosetta-style stub
	
	>>> print stub( vec(0,0,0), vec(-1,0,0), vec(0,1,0) )            #doctest: +NORMALIZE_WHITESPACE
	XFORM:
	    MAT(1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,1.000000 )
	    VEC(0.000000,0.000000,0.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	assert isinstance(c,VEC)
	x = normalizedv(subvv(a,b));
	z = normalizedv(crossvv(x,subvv(c,b)));
	y = crossvv(z,x);
	s = XFORM();
	s.R = cols(x,y,z);
	s.t = a;
	return s;

def stubrev(a, b, c):
	"""get a global-to-local stub: stubrev(x) == xrev(stub(x))
	
	>>> xnull = multxx( stubrev(vec(1,2,3),vec(3,2,1),vec(2,1,3)), stub(vec(1,2,3),vec(3,2,1),vec(2,1,3)) )
	>>> print xnull                    #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,1.000000 )
		VEC(0.000000,0.000000,0.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	assert isinstance(c,VEC)
	x = normalizedv(subvv(a,b));
	z = normalizedv(crossvv(x,subvv(c,b)));
	y = crossvv(z,x);
	s = XFORM();
	s.R = rows(x,y,z);
	s.t = multmv(s.R,vec(-a.x,-a.y,-a.z));
	return s;

def stubc(cen, a, b, c):
	"""stub creation with explicit center (rather than a): stub(a,b,c)==stubc(a,a,b,c)
	
	>>> print  stubc(vec(0,0,0), vec(1,2,3), vec(3,2,1), vec(2,1,3))     #doctest: +NORMALIZE_WHITESPACE
	XFORM:
    	MAT(-0.707107,0.408248,0.577350 0.000000,-0.816497,0.577350 0.707107,0.408248,0.577350 )
    	VEC(0.000000,0.000000,0.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	assert isinstance(c,VEC)
	assert isinstance(cen,VEC)
	x = normalizedv(subvv(a,b));
	z = normalizedv(crossvv(x,subvv(c,b)));
	y = crossvv(z,x);
	s = XFORM();
	s.R = cols(x,y,z);
	s.t = cen;
	return s;

def stubcrev(cen, a, b, c):
	"""global-to-local stub creation with explicit center (rather than a): stubrev(a,b,c)==stubcrev(a,a,b,c)
	
	>>> print  stubcrev(vec(0,0,0), vec(1,2,3), vec(3,2,1), vec(2,1,3))     #doctest: +NORMALIZE_WHITESPACE
	XFORM:
    	MAT(-0.707107,0.000000,0.707107 0.408248,-0.816497,0.408248 0.577350,0.577350,0.577350 )
    	VEC(0.000000,0.000000,-0.000000)
	>>> x  = stubc   (vec(0,0,0), vec(1,2,3), vec(3,2,1), vec(2,1,3))
	>>> xr = stubcrev(vec(0,0,0), vec(1,2,3), vec(3,2,1), vec(2,1,3))
	>>> print multxx(x,xr)                                                  #doctest: +NORMALIZE_WHITESPACE
	XFORM:
    	MAT(1.000000,0.000000,0.000000 0.000000,1.000000,0.000000 0.000000,0.000000,1.000000 )
    	VEC(0.000000,0.000000,0.000000)
	"""
	assert isinstance(a,VEC)
	assert isinstance(b,VEC)
	assert isinstance(c,VEC)
	assert isinstance(cen,VEC)
	x = normalizedv(subvv(a,b));
	z = normalizedv(crossvv(x,subvv(c,b)));
	y = crossvv(z,x);
	s = XFORM();
	s.R = rows(x,y,z);
	s.t = multmv(s.R,vec(-cen.x,-cen.y,-cen.z));
	return s;


def nparray_to_VECs(ary):
	"""
	>>> import numpy as np
	>>> ary = np.arange(12)
	>>> for v in nparray_to_VECs(ary): print v
	VEC(0.000000,1.000000,2.000000)
	VEC(3.000000,4.000000,5.000000)
	VEC(6.000000,7.000000,8.000000)
	VEC(9.000000,10.000000,11.000000)
	"""
	if len(ary.shape)==1: tmp = ary.shape = (len(ary)/3,3)
	assert len(ary.shape)==2 and ary.shape[1]==3
	vecs = []
	for i in range(len(ary)):
		vecs.append(vec(*ary[i]))
	return vecs

def nparray_to_MATs(ary):
	"""
	>>> import numpy as np
	>>> ary = np.arange(18)
	>>> for m in nparray_to_MATs(ary): print m
	MAT(0.000000,1.000000,2.000000 3.000000,4.000000,5.000000 6.000000,7.000000,8.000000 )
	MAT(9.000000,10.000000,11.000000 12.000000,13.000000,14.000000 15.000000,16.000000,17.000000 )
	"""
	if len(ary.shape)==1: tmp = ary.shape = (len(ary)/9,9)
	assert len(ary.shape)==2 and ary.shape[1]==9
	mats = []
	for i in range(len(ary)):
		mats.append(rowsf(*ary[i]))
	return mats

def nparray_to_XFORMs(ary):
	"""
	>>> import numpy as np
	>>> ary = np.arange(24)
	>>> for m in nparray_to_XFORMs(ary): print m       #doctest: +NORMALIZE_WHITESPACE
	XFORM:
		MAT(0.000000,1.000000,2.000000 3.000000,4.000000,5.000000 6.000000,7.000000,8.000000 )
		VEC(9.000000,10.000000,11.000000)
	XFORM:
		MAT(12.000000,13.000000,14.000000 15.000000,16.000000,17.000000 18.000000,19.000000,20.000000 )
		VEC(21.000000,22.000000,23.000000)
	"""
	if len(ary.shape)==1: tmp = ary.shape = (len(ary)/12,12)
	assert len(ary.shape)==2 and ary.shape[1]==12
	xfs = []
	for i in range(len(ary)):
		xfs.append(xform(rowsf(*ary[i][:9]),vec(*ary[i][9:])))
	return xfs


if __name__ == '__main__':
	import doctest
	tr = doctest.testmod()
	print "tests passed:",tr.attempted-tr.failed
	print "tests failed:",tr.failed
