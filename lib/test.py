import sys
import pyopencl as cl
import pyopencl.clrandom as clrand
import numpy as np
import numpy.linalg as la

def get_exposed_atoms(pose):
	r.core.id.AtomID_Map

def get_gpu_context():
	platforms = cl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==cl.device_type.GPU]
	return cl.Context(gpus)
	


KERN = """
#define RANLUXCL_LUX 2 // 0-4, lower=faster, higher=better

#include <pyopencl-ranluxcl.cl>

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

__kernel void test(
	uint seed,
	__global ranluxcl_state_t *rldat,
	__global float *output,
	int out_size
){
	myrand_init(seed,rldat);

	int ngroup = get_global_size(1);
	out_size = out_size / ngroup;

	for(int i = 0; i < out_size/4; ++i){
		__local float4 rand4;
		if(get_local_id(0) == 0) {
			rand4 = myrand(rldat);
			output[ get_global_id(1)*out_size + 4*i + 0] = rand4.x;
			output[ get_global_id(1)*out_size + 4*i + 1] = rand4.y;
			output[ get_global_id(1)*out_size + 4*i + 2] = rand4.z;
			output[ get_global_id(1)*out_size + 4*i + 3] = rand4.w;						
		}
	}

}

"""

def demo():
	a = np.random.rand(64,64).astype(np.float32)
	b = np.random.rand(64,64).astype(np.float32)
	
	ctx = get_gpu_context()
	queue = cl.CommandQueue(ctx)
	
	mf = cl.mem_flags
	a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
	dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)
	
	prg = cl.Program(ctx, """
	    __kernel void sum(__global const float *a,
	    __global const float *b, __global float *c)
	    {
	      int gid = get_global_id(0);
	      c[gid] = a[gid] + b[gid];
	    }
		""").build()
	
	prg.sum(queue, (64,64), (64,1), a_buf, b_buf, dest_buf)
	
	a_plus_b = np.empty_like(a)
	cl.enqueue_copy(queue, a_plus_b, dest_buf)
	
	print la.norm(a_plus_b - (a+b))


def testlocalsize(d1,d2,d3):
	ctx = get_gpu_context()
	queue = cl.CommandQueue(ctx)
	
	ary = cl.array.empty(queue, (d1*d2*d3,), dtype=np.int32)
	
	prg = cl.Program(ctx, """
	#define  gli0 get_local_id(0)
	#define  gli1 get_local_id(1)
	#define  gli2 get_local_id(2)
	#define  gls0 get_local_size(0)
	#define  gls1 get_local_size(1)
	#define  gls2 get_local_size(2)
	    __kernel void test(
			__global int *output)
	    {
			int i = gli0 + gli1*gls0 + gli2*gls0*gls1;
			output[i] = i;
	    }
		""").build()
	
	kernel = prg.test
	#print kernel.get_work_group_info(cl.kernel_work_group_info.COMPILE_WORK_GROUP_SIZE)
	kernel(queue, (d1,d2,d3), (d1,d2,d3), ary.data )
	
	print ary


if __name__ == '__main__':
	#testlocalsize(10,2,3);
	#demo()

	ctx = get_gpu_context()
	queue = cl.CommandQueue(ctx)

	# rg = clrand.RanluxGenerator(queue=queue,num_work_items=10,max_work_items=10,luxury=2)
	# rgkernel = rg.get_gen_kernel(np.float32)
	# print rg.uniform(queue,(4,4),np.float32)

	prg = cl.Program(ctx,KERN).build()
	knl = prg.test
	knl.set_scalar_arg_dtypes([np.uint32, None, None, np.int32])
	
	state = cl.array.empty(queue, (1, 112), dtype=np.uint8)
	state.fill(17)
	
	ary = cl.array.empty(queue, (128,), dtype=np.float32)
	ary += 1.2345
	
	#knl(queue, (64,8), (64,1), 420565, state.data, ary.data, ary.size )
	knl(queue, (64,2), (64,1), 420565, state.data, ary.data, ary.size )
	
	print ary
	s = set(ary.get())
	assert len(ary) == len(s)
	
	
	