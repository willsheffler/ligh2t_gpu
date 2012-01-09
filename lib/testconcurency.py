import sys,os,pyopencl,pyopencl,numpy,pyopencl.array,random,time
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
// #define GI0 get_global_id(0)
#define GI1 get_global_id(1)
// #define GI2 get_global_id(2)
#define LI0 get_local_id(0)
// #define LI1 get_local_id(1)
// #define LI2 get_local_id(2)
// #define GS0 get_global_size(0)
// #define GS1 get_global_size(1)
// #define GS2 get_global_size(2)
#define LS0 get_local_size(0)
// #define LS1 get_local_size(1)
// #define LS2 get_local_size(2)

#define RANLUXCL_LUX 4 // 0-4, lower=faster, higher=better

void myrand_init(uint seed, __global ranluxcl_state_t *ranluxcltab);
float4 myrand(__global ranluxcl_state_t *ranluxcltab);

void myrand_init(uint seed, __global ranluxcl_state_t *ranluxcltab) {
	ranluxcl_initialization(seed, ranluxcltab);
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
	__global uint *python_seed,
	__global struct VEC   *out,
	__global struct XFORM *xout,
	__global float *rout
){

	if(LI0==0) myrand_init(GI1+*python_seed,ranluxcltab);

	__local struct VEC lkh[nlkh],lkp[nlkp];
	for(size_t i = 0; i < nlkh; i+=LS0) if((i+LI0)<nlkh) lkh[i+LI0]=init_lkh[i+LI0];
	for(size_t i = 0; i < nlkp; i+=LS0) if((i+LI0)<nlkp) lkp[i+LI0]=init_lkp[i+LI0];
  	barrier(CLK_LOCAL_MEM_FENCE);

	__local struct XFORM xh,xp;
	if(LI0==0) xh = multxx( stub(lkh[nlkh-3],lkh[nlkh-2],lkh[nlkh-1]) , hyd_stub );
	if(LI0==0) xp = multxx( stub(lkp[nlkp-3],lkp[nlkp-2],lkp[nlkp-1]) , pet_stub );
  	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(int ITER = 0; ITER < 100; ++ITER) {
		__local uint nlnk,idx,residx;
		__local struct XFORM move;
		__local bool p_or_h;
      	//barrier(CLK_LOCAL_MEM_FENCE);
		if(LI0==0) {
			float4 rand = myrand(ranluxcltab);
			p_or_h = rand.x < 0.5;
			residx = (uint)(rand.y*(float)(p_or_h?nlkp/3:nlkh/3)-2.0) + 1;
			bool phipsi = rand.z < 0.5;
			float ang = rand.w*6.283185;
			idx = 3*residx + (phipsi?0:1);
			nlnk = p_or_h?nlkp:nlkh; 
			struct VEC cen = (p_or_h?lkp:lkh)[idx];
			struct VEC axs = subvv( (p_or_h?lkp:lkh)[idx+1], (p_or_h?lkp:lkh)[idx] );
			struct MAT R = rotation_matrix( axs, ang );
			move = xform( R, subvv(cen,multmv(R,cen)) );
			if(p_or_h) xp = multxx(move,xp);
			else       xh = multxx(move,xh);
			rout[GI1] = rand.x;
			//for(uint i = idx+2; i < nlnk; ++i) (p_or_h?lkp:lkh)[i] = multxv(move,(p_or_h?lkp:lkh)[i]);
		}
      	//barrier(CLK_LOCAL_MEM_FENCE);
		for(uint i = idx+2; i < nlnk; i+=LS0) {
			if(LI0+i < nlnk) (p_or_h?lkp:lkh)[LI0+i] = multxv(move,(p_or_h?lkp:lkh)[LI0+i]);
		}
		// if( idx+1 < (      LI0) && (      LI0) < nlnk ) (p_or_h?lkp:lkh)[      LI0] = multxv(move,(p_or_h?lkp:lkh)[      LI0]);
		// if( idx+1 < (  LS0+LI0) && (  LS0+LI0) < nlnk ) (p_or_h?lkp:lkh)[  LS0+LI0] = multxv(move,(p_or_h?lkp:lkh)[  LS0+LI0]);
		// if( idx+1 < (2*LS0*LI0) && (2*LS0*LI0) < nlnk ) (p_or_h?lkp:lkh)[2*LS0*LI0] = multxv(move,(p_or_h?lkp:lkh)[2*LS0*LI0]);
		
	}
	
	//barrier(CLK_LOCAL_MEM_FENCE);
	//for(size_t i=0; i<npsi; i+=LS0) if((i+LI0)<npsi) out[GI1*ntot+                    i+LI0] =           psi[i+LI0];
	//for(size_t i=0; i<nlkh; i+=LS0) if((i+LI0)<nlkh) out[GI1*ntot+npsi               +i+LI0] =           lkh[i+LI0];
	//for(size_t i=0; i<nhyd; i+=LS0) if((i+LI0)<nhyd) out[GI1*ntot+npsi+nlkh          +i+LI0] = multxv(xh,hyd[i+LI0]);
	//for(size_t i=0; i<nlkp; i+=LS0) if((i+LI0)<nlkp) out[GI1*ntot+npsi+nlkh+nhyd     +i+LI0] =           lkp[i+LI0];
	//for(size_t i=0; i<npet; i+=LS0) if((i+LI0)<npet) out[GI1*ntot+npsi+nlkh+nhyd+nlkp+i+LI0] = multxv(xp,pet[i+LI0]);
	//for(size_t i=0; i<nlkh; i+=LS0) if((i+LI0)<nlkh) out[GI1*ntot               +i+LI0] =           lkh[i+LI0];
	//for(size_t i=0; i<nhyd; i+=LS0) if((i+LI0)<nhyd) out[GI1*ntot+nlkh          +i+LI0] = multxv(xh,hyd[i+LI0]);
	//for(size_t i=0; i<nlkp; i+=LS0) if((i+LI0)<nlkp) out[GI1*ntot+nlkh+nhyd     +i+LI0] =           lkp[i+LI0];
	//for(size_t i=0; i<npet; i+=LS0) if((i+LI0)<npet) out[GI1*ntot+nlkh+nhyd+nlkp+i+LI0] = multxv(xp,pet[i+LI0]);
	for(size_t i=0; i<nlkh; i+=LS0) if((i+LI0)<nlkh) out[GI1*ntot     +i+LI0] = lkh[i+LI0];
	for(size_t i=0; i<nlkp; i+=LS0) if((i+LI0)<nlkp) out[GI1*ntot+nlkh+i+LI0] = lkp[i+LI0];

	if(LI0==0) {
		xout[3*GI1+0] = xh;
		xout[3*GI1+1] = xp;
		xout[3*GI1+2] = multxx( xrev(xh), xp ); //pet to hyd
	}
	
}
"""%vars()
for n in "npsi nhyd npet nlkp nlkh ntot".split(): KERN = KERN.replace(n,'%('+n+')i')
for n in "hyd_stub pet_stub".split(): KERN = KERN.replace(n,'%('+n+')s')

def process_results(out,xout,rout,NG):
	print "NUM 0 coords:",sum(out.get()==0.0)
	if int(os.popen("diff ../test/test1.pdb ../test/test2.pdb | wc -l").read()):
		  print "OUTPUTS DO NOT MATCH!!!!!!!!!!"
	else: print "OUTPUTS MATCH!"
	if int(os.popen("diff ../test/test1.pdb ../test/test1.pdb.last | wc -l").read()):
		  print "OUTPUT DOES NOT MATCH PREVIOUS!!!!!!!!!!"
	else: print "PREVIOUS OUTPUT MATCHS!"
	ary = out.get()
	for i in range(NG):
		dump_pdb(ary[i*len(ary)/NG:(i+1)*len(ary)/NG],"../test/GI_%i.pdb"%i)

def test():
	NL,NG = 64,1024
	ctx,queue = my_get_gpu_context()
	# inputs
	components = "psi hyd pet len ldc".split()
	coords = [xyzHash("../dat/%s.xyzHash"%n).get_coord_clarray(queue) for n in components]
	if   components[3][-1]=="c": hyd_stub = "stubrev(vec"+str(hyd_N_N)+",vec"+str(hyd_N_CA)+",vec"+str(hyd_N_C)+")"
	elif components[3][-1]=="n": hyd_stub = "stubrev(vec"+str(hyd_C_C)+",vec"+str(hyd_C_CA)+",vec"+str(hyd_C_N)+")"
	else: assert "CAN'T DETERMINE ORIENTATIN OF HYDA1 LINKER!!!"==None
	if   components[4][-1]=="c": pet_stub = "stubrev(vec"+str(pet_N_N)+",vec"+str(pet_N_CA)+",vec"+str(pet_N_C)+")"
	elif components[4][-1]=="n": pet_stub = "stubrev(vec"+str(pet_C_C)+",vec"+str(pet_C_CA)+",vec"+str(pet_C_N)+")"
	else: assert "CAN'T DETERMINE ORIENTATIN OF HYDA1 LINKER!!!"==None	
	npsi,nhyd,npet = (coords[i].size/3 for i in (0,1,2))
	lh = 50
	lp = 50
	nlkh,nlkp = lh*3,lp*3
	#ntot = npsi+nhyd+npet+nlkh+nlkp
	ntot = nlkh+nlkp
	ranlux = pyopencl.array.empty(queue, (112*NG,), dtype=numpy.int8)
	#print python_seed
	# outputs
	out    = pyopencl.array.empty(queue, (3*ntot*NG,), dtype=numpy.float32)
	xout   = pyopencl.array.empty(queue, (12*3*NG,)  , dtype=numpy.float32)
	rout   = pyopencl.array.empty(queue, (NG,)       , dtype=numpy.float32)
	seed   = pyopencl.array.empty(queue, (1,)        , dtype=numpy.uint32)
	kargs = [x.data for x in coords+[ranlux,seed,out,xout,rout] ]
	my_test_kernel = pyopencl.Program(ctx,KERN%vars()).build(OPTS).my_test_kernel
	for ITER in range(100):
		seed.fill(random.getrandbits(31))
		ranlux.fill(17)		
		my_test_kernel(queue,(NL,NG),(NL,1), *kargs ).wait()#; print seed.get(),rout
		lastout  = out.get()
		lastxout = xout.get()
		ranlux.fill(17)		
		my_test_kernel(queue,(NL,NG),(NL,1), *kargs ).wait()#; print seed.get(),rout
		assert numpy.all(lastxout == xout.get())
		assert numpy.all(lastout  ==  out.get())
		assert numpy.sum(numpy.abs(out.get()[:-3]-out.get()[3:])>1.6)  < 3*2*NG
		print "1 vs 64 threads same! seed:",seed.get()[0]
	# for i in range(NG):
	# 	dump_pdb(ary[i*len(ary)/NG:(i+1)*len(ary)/NG],"../test/test%i_%i_%i.pdb"%(i,lh,lp))
	# 	xform_pdb(nparray_to_XFORMs(xout.get())[3*i+0],"../pdb/hyda1_hash_bb.pdb","../test/.tmp")
	# 	os.system("cat ../test/.tmp >> ../test/test%i_%i_%i.pdb"%(i,lh,lp))
	# 	xform_pdb(nparray_to_XFORMs(xout.get())[3*i+1],"../pdb/petf_hash_bb.pdb" ,"../test/.tmp")
	# 	os.system("cat ../test/.tmp >> ../test/test%i_%i_%i.pdb"%(i,lh,lp))
	# 	#xform_pdb(nparray_to_XFORMs(xout.get())[2],"../pdb/petf_hash_bb.pdb" ,"../test/petf_xform_hyda1.pdb")
	# 	#process_results(out,xout,rout,NG)
	# 	#print out
	# 	#print rout, lh,lp
		


if __name__ == '__main__':
	test()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	