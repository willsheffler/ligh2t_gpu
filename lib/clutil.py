import pyopencl,numpy,sys

def my_get_gpu_context():
	platforms = pyopencl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==pyopencl.device_type.GPU]
	#print gpus[0].get_info(pyopencl.device_info.EXTENSIONS)
	i = int(sys.argv[1])
	print "using GPU #",i
	gpus = [ gpus[i] ]
	ctx = pyopencl.Context(gpus)
	return ctx,pyopencl.CommandQueue(ctx)

def my_get_cpu_context():
	platforms = pyopencl.get_platforms()
	assert len(platforms) == 1
	platform = platforms[0]
	gpus = [d for d in platform.get_devices() if d.type==pyopencl.device_type.CPU]
	ctx = pyopencl.Context(gpus)
	return ctx,pyopencl.CommandQueue(ctx)


MATHOPTS = "-cl-single-precision-constant -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math -w"


def points_cen(x):
	return numpy.array([numpy.mean(x[:,i]) for i in (0,1,2)])
