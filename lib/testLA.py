import sys
import pyopencl as cl
import pyopencl.clrandom as clrand
import numpy as np
import numpy.linalg as la
from xyzHash import xyzHash

def get_gpu_context():
	platforms = cl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==cl.device_type.GPU]
	return cl.Context(gpus)

def get_cpu_context():
	platforms = cl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==cl.device_type.CPU]
	return cl.Context(gpus)


KERN = """
#include <cl/gpu_mat_vec.cl>
//#include </Users/sheffler/project/mosetta/prototypes/ligh2t/lib/cl/gpu_mat_vec.cl>


__kernel void test(
	__global struct MAT *vecs,
	__global float *output
){
output[0] = vecs[0].xx;
output[1] = vecs[0].xy;
output[2] = vecs[0].xz;
output[3] = vecs[0].yx;
output[4] = vecs[0].yy;
output[5] = vecs[0].yz;
output[6] = vecs[0].zx;
output[7] = vecs[0].zy;
output[8] = vecs[0].zz;
output[10] = vecs[3].xx;
output[11] = vecs[3].xy;
output[12] = vecs[3].xz;
output[13] = vecs[3].yx;
output[14] = vecs[3].yy;
output[15] = vecs[3].yz;
output[16] = vecs[3].zx;
output[17] = vecs[3].zy;
output[18] = vecs[3].zz;
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
	ctx = get_gpu_context()
	queue = cl.CommandQueue(ctx)

	prg = cl.Program(ctx,KERN)
	prg.build(options="-I/Users/sheffler/project/mosetta/prototypes/ligh2t/lib -cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -w")
	knl = prg.test
	knl.set_scalar_arg_dtypes([None, None])
		
	ary = cl.array.empty(queue, (90,), dtype=np.float32)
	out = cl.array.empty(queue, (90,), dtype=np.float32)
	ary.set(np.arange(90,dtype=np.float32))

	knl(queue, (1,1), (1,1) , ary.data, out.data )
	
	print out

	
	