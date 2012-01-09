import sys,os,pyopencl,pyopencl,numpy,pyopencl.array,random
import gpu_mat_vec as gmv
from xyzHash import xyzHash
from struct_data import *
from pdbutil import xform_pdb,dump_pdb
from gpu_mat_vec import nparray_to_XFORMs

def my_get_gpu_context():
	platforms = pyopencl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==pyopencl.device_type.GPU]
	#print gpus[0].get_info(pyopencl.device_info.EXTENSIONS)
	ctx = pyopencl.Context(gpus)
	return ctx,pyopencl.CommandQueue(ctx)

def my_get_cpu_context():
	platforms = pyopencl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==pyopencl.device_type.CPU]
	ctx = pyopencl.Context(gpus)
	return ctx,pyopencl.CommandQueue(ctx)



OPTS = "-I/Users/sheffler/project/mosetta/prototypes/ligh2t/lib -cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -w"
UTILITY_FUNCS = """
#define GI0 get_global_id(0)
#define GI1 get_global_id(1)
#define GI2 get_global_id(2)
#define LI0 get_local_id(0)
#define LI1 get_local_id(1)
#define LI2 get_local_id(2)
#define GS0 get_global_size(0)
#define GS1 get_global_size(1)
#define GS2 get_global_size(2)
#define LS0 get_local_size(0)
#define LS1 get_local_size(1)
#define LS2 get_local_size(2)

#define RANLUXCL_LUX 2 // 0-4, lower=faster, higher=better

void myrand_init(uint seed, __global ranluxcl_state_t *ranluxcltab);
float4 myrand(__global ranluxcl_state_t *ranluxcltab);

void myrand_init(uint seed, __global ranluxcl_state_t *ranluxcltab) {
	int ID = get_global_id(0)+get_global_id(1)*get_global_size(0)+get_global_id(2)*get_global_size(0)*get_global_size(1);
	ranluxcl_initialization(seed+ID, ranluxcltab);
}

float4 myrand(__global ranluxcl_state_t *ranluxcltab) {
	float4 rand4;
	ranluxcl_state_t ranluxclstate;
	ranluxcl_download_seed(&ranluxclstate, ranluxcltab); // get from global mem
	rand4 = ranluxcl32(&ranluxclstate);
	ranluxcl_upload_seed(&ranluxclstate, ranluxcltab); // store to global mem
	return rand4;
}

"""
OUTPUT_COORDS = """
	for(size_t i=0; i<npsi; i+=LS0) if((i+LI0)<npsi) out[GI1*ntot+                    i+LI0] =           psi[i+LI0];
	for(size_t i=0; i<nlkh; i+=LS0) if((i+LI0)<nlkh) out[GI1*ntot+npsi               +i+LI0] =           lkh[i+LI0];
	for(size_t i=0; i<nhyd; i+=LS0) if((i+LI0)<nhyd) out[GI1*ntot+npsi+nlkh          +i+LI0] = multxv(xh,hyd[i+LI0]);
	for(size_t i=0; i<nlkp; i+=LS0) if((i+LI0)<nlkp) out[GI1*ntot+npsi+nlkh+nhyd     +i+LI0] =           lkp[i+LI0];
	for(size_t i=0; i<npet; i+=LS0) if((i+LI0)<npet) out[GI1*ntot+npsi+nlkh+nhyd+nlkp+i+LI0] = multxv(xp,pet[i+LI0]);
	if(GI0==0) xout[0] = xh;
	if(GI0==0) xout[1] = xp;
"""
KERN = """
#include <pyopencl-ranluxcl.cl>
#include <cl/gpu_mat_vec.cl>

%(UTILITY_FUNCS)s

__kernel void my_test_kernel(
	__constant struct VEC const *psi,
	__constant struct VEC const *hyd,
	__constant struct VEC const *pet,
	__global   struct VEC const *init_lkh,
	__global   struct VEC const *init_lkp,
	__global ranluxcl_state_t *ranluxcltab,
	__global struct VEC   *out,
	__global struct XFORM *xout,
	__global float *rout
){
	myrand_init(GI1+python_seed,ranluxcltab);
	rout[GI1] = myrand(ranluxcltab).x;
	
	__local struct VEC lkh[nlkh],lkp[nlkp];
	for(size_t i = 0; i < nlkh; i+=LS0) if((i+LI0)<nlkh) lkh[i+LI0]=init_lkh[i+LI0];
	for(size_t i = 0; i < nlkp; i+=LS0) if((i+LI0)<nlkp) lkp[i+LI0]=init_lkp[i+LI0];

	__local struct XFORM xh,xp;
	if(LI0==0) xh = multxx( stub(lkh[nlkh-3],lkh[nlkh-2],lkh[nlkh-1]) , hyd_stub );
	if(LI0==0) xp = multxx( stub(lkp[nlkp-3],lkp[nlkp-2],lkp[nlkp-1]) , pet_stub );
	
	%(OUTPUT_COORDS)s
}
"""%vars()
for n in "npsi nhyd npet nlkp nlkh ntot python_seed".split(): KERN = KERN.replace(n,'%('+n+')i')
for n in "hyd_stub pet_stub".split(): KERN = KERN.replace(n,'%('+n+')s')

def process_results(out,xout,rout,Nglobal):
	print "NUM 0 coords:",sum(out.get()==0.0)
	if int(os.popen("diff ../test/test1.pdb ../test/test2.pdb | wc -l").read()):
		  print "OUTPUTS DO NOT MATCH!!!!!!!!!!"
	else: print "OUTPUTS MATCH!"
	if int(os.popen("diff ../test/test1.pdb ../test/test1.pdb.last | wc -l").read()):
		  print "OUTPUT DOES NOT MATCH PREVIOUS!!!!!!!!!!"
	else: print "PREVIOUS OUTPUT MATCHS!"
	xform_pdb(nparray_to_XFORMs(xout.get())[0],"../pdb/hyda1_hash.pdb","../test/hyda1_xform.pdb")
	xform_pdb(nparray_to_XFORMs(xout.get())[1],"../pdb/petf_hash.pdb" ,"../test/petf_xform.pdb")
	print rout
	ary = out.get()
	for i in range(Nglobal):
		dump_pdb(ary[i*len(ary)/Nglobal:(i+1)*len(ary)/Nglobal],"../test/GI_%i.pdb"%i)

def test():
	Nlocal  = 64
	Nglobal = 1024
	
	ctx,queue = my_get_gpu_context()
	
	# inputs
	components = "psi hyd pet ldn lec".split()
	coords = [xyzHash("../dat/%s.xyzHash"%n).get_coord_clarray(queue) for n in components]
	if   components[3][-1]=="c": hyd_stub = "stubrev(vec"+str(hyd_N_N)+",vec"+str(hyd_N_CA)+",vec"+str(hyd_N_C)+")"
	elif components[3][-1]=="n": hyd_stub = "stubrev(vec"+str(hyd_C_C)+",vec"+str(hyd_C_CA)+",vec"+str(hyd_C_N)+")"
	else: assert "CAN'T DETERMINE ORIENTATIN OF HYDA1 LINKER!!!"==None
	if   components[4][-1]=="c": pet_stub = "stubrev(vec"+str(pet_N_N)+",vec"+str(pet_N_CA)+",vec"+str(pet_N_C)+")"
	elif components[4][-1]=="n": pet_stub = "stubrev(vec"+str(pet_C_C)+",vec"+str(pet_C_CA)+",vec"+str(pet_C_N)+")"
	else: assert "CAN'T DETERMINE ORIENTATIN OF HYDA1 LINKER!!!"==None	
	npsi,nhyd,npet = (coords[i].size/3 for i in (0,1,2))
	nlkh,nlkp = 23*3,32*3
	ntot = npsi+nhyd+npet+nlkh+nlkp
	ranlux = pyopencl.array.empty(queue, (112*Nglobal,), dtype=numpy.int8); ranlux.fill(17)
	python_seed = random.getrandbits(31)
	# outputs
	out    = pyopencl.array.empty(queue, (3*ntot*Nglobal,), dtype=numpy.float32)
	print out.size*4/1024/1024
	xout   = pyopencl.array.empty(queue, (12*10,)       , dtype=numpy.float32)
	rout   = pyopencl.array.empty(queue, (Nglobal,)    , dtype=numpy.float32)
	# run
	kargs = [x.data for x in coords+[ranlux,out,xout,rout] ]
	if os.path.exists("../test/test1.pdb"): os.rename("../test/test1.pdb","../test/test1.pdb.last")
	print "build"
	my_test_kernel = pyopencl.Program(ctx,KERN%vars()).build(OPTS).my_test_kernel
	# print "run 1"
	# my_test_kernel(queue,(   1  ,Nglobal),(   1  ,1), *kargs );	dump_pdb(out.get()[0:],"../test/test1.pdb")
	print "run 2"
	my_test_kernel(queue,(Nlocal,Nglobal),(Nlocal,1), *kargs );	dump_pdb(out.get()[0:],"../test/test2.pdb")
	# print "output results"
	# process_results(out,xout,rout,Nglobal)

if __name__ == '__main__':
	test()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	