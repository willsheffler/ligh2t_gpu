#include <OpenCL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>


void clCheckError( cl_int clErrNum, std::string const & message )
{
  if ( clErrNum != CL_SUCCESS ) {
    std::cerr << "Error: " << message << ", err= " << clErrNum << std::endl;
    exit(1);
  }
}

int main()
{
  cl_platform_id cpPlatform;       //OpenCL platform
  cl_device_id cdDevice;           //OpenCL devices
  cl_context      cxGPUContext;    //OpenCL context
  cl_command_queue cqCommandQueue; //OpenCL command que

  char chBuffer[1024];
  cl_uint num_platforms;
  cl_platform_id* clPlatformIDs;
  cl_int ciErrNum;

  // Get OpenCL platform count
  ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
  clCheckError( ciErrNum, "clGetPlatformIDs" );
  std::cout << "num_platforms " <<  num_platforms << std::endl;

  if(num_platforms == 0) {
    std::cerr << "no platforms!" << std::endl;
    exit(1);
  } else {
    // if there's a platform or more, make space for ID's                                                                                                                              
    if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL) {
      std::cerr << "failed to allocate memory for clPlatformIDs" << std::endl;
      exit(1);
    }

    // get platform info for each platform and trap the NVIDIA platform if found                                                                                                       
    ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
    clCheckError( ciErrNum, "clGetPlatformIDs" );
    cpPlatform = clPlatformIDs[0]; // take the first platform
    free( clPlatformIDs );
  }

  // Get a GPU device                                                                                                                                                                    
  ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
  clCheckError(ciErrNum, "clGetDeviceIDs" );

  // Create the context                                                                                                                                                                  
  cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErrNum);
  clCheckError(ciErrNum, "clCreateContext" );

  //Create a command-queue                                                                                                                                                               
  cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
  clCheckError(ciErrNum, "clCreateCommandQueue");

  cl_program cpSum;   //OpenCL program                                                                                                                                           
  cl_kernel  ckSum;   //OpenCL kernel                                                                                                                                            

  // OK let's compile the program
	std::cout << "reading kernels from: /Users/sheffler/project/mosetta/prototypes/ligh2t/test/test.cl" << std::endl;
  std::ifstream kin("/Users/sheffler/project/mosetta/prototypes/ligh2t/test/test.cl");
  std::string src((std::istreambuf_iterator<char>(kin)), std::istreambuf_iterator<char>());

  size_t kernelLength = src.size();
  char ** program_array_of_strings = new char*[ 1 ];
  program_array_of_strings[0] = new char[ kernelLength ];
  memcpy( program_array_of_strings[0], src.c_str(),  sizeof(char) * kernelLength );
  cpSum = clCreateProgramWithSource(cxGPUContext, 1, (const char **) program_array_of_strings, &kernelLength, &ciErrNum);
  delete [] program_array_of_strings[0];
  delete [] program_array_of_strings;
  clCheckError( ciErrNum, "clCreateProgramWithSource");

////////////////////// will /////////////////////
	{
		using namespace std;
    cout << "clBuildProgram" << endl;
	  clBuildProgram(cpSum, 0, NULL, "", NULL, &ciErrNum);
	  //if(ciErrNum != CL_SUCCESS) {
	    size_t len_status, len_options, len_log;
	    char buffer_options[65535];
	    char buffer_log    [65535];
	    cout << "Error: Failed to build program executable!" << endl;

	    cl_build_status status;
	    clGetProgramBuildInfo(cpSum, cdDevice, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status , &len_status);
	    cout << "CL_PROGRAM_BUILD_STATUS: '" << status << "'" << endl;

	    clGetProgramBuildInfo(cpSum, cdDevice, CL_PROGRAM_BUILD_OPTIONS, sizeof(buffer_options), buffer_options, &len_options);
	    cout << "CL_PROGRAM_BUILD_OPTIONS: '";
	    for(int i=0;i<len_options;++i) cout << buffer_options[i];
	    cout << "'" << endl;

	    clGetProgramBuildInfo(cpSum, cdDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer_log), buffer_log, &len_log);
	    cout << "---- start build log ----" << endl;
	    for(int i=0;i<len_log;++i) cout << buffer_log[i];
	    cout << endl << "----  end build log  ----" << endl;

		  clCheckError( ciErrNum, "clBuildProgram");
	  //}
	}
////////////////////// end will //////////////////




  ckSum = clCreateKernel( cpSum, "sum_two_into_one", &ciErrNum );
  clCheckError( ciErrNum, "clCreateKernel sum" );


  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;
  d_a = clCreateBuffer( cxGPUContext, CL_MEM_READ_ONLY, 128 * sizeof( float ), NULL, &ciErrNum );
  clCheckError( ciErrNum, "clCreateBuffer d_a" );
  d_b = clCreateBuffer( cxGPUContext, CL_MEM_READ_ONLY, 128 * sizeof( float ), NULL, &ciErrNum );
  clCheckError( ciErrNum, "clCreateBuffer d_b" );
  d_c = clCreateBuffer( cxGPUContext, CL_MEM_WRITE_ONLY, 128 * sizeof( float ), NULL, &ciErrNum );
  clCheckError( ciErrNum, "clCreateBuffer d_b" );

  std::vector< float > h_a( 128, 7.5f ), h_b( 128, 8.5f ), h_c( 128, 0.0f );
  ciErrNum = clEnqueueWriteBuffer( cqCommandQueue, d_a, CL_FALSE, 0, 128 * sizeof( float ), & h_a[ 0 ], 0, NULL, NULL );
  clCheckError( ciErrNum, "clEnqueuWriteBuffer d_a" );
  ciErrNum = clEnqueueWriteBuffer( cqCommandQueue, d_b, CL_FALSE, 0, 128 * sizeof( float ), & h_b[ 0 ], 0, NULL, NULL );
  clCheckError( ciErrNum, "clEnqueuWriteBuffer d_b" );

  std::cout << "setting kernel arguments" << std::endl;
  ciErrNum = clSetKernelArg( ckSum, 0, sizeof( float * ), &d_a );
  clCheckError( ciErrNum, "clSetKernelArg d_a" );
  ciErrNum = clSetKernelArg( ckSum, 1, sizeof( float * ), &d_b );
  clCheckError( ciErrNum, "clSetKernelArg d_b" );
  ciErrNum = clSetKernelArg( ckSum, 2, sizeof( float * ), &d_c );
  clCheckError( ciErrNum, "clSetKernelArg d_c" );
  int globalsize( 128 );
  ciErrNum = clSetKernelArg( ckSum, 3, sizeof( int ), &globalsize );
  clCheckError( ciErrNum, "clSetKernelArg 128" );

  std::cout << "about to launch kernel" << std::endl;
  std::vector< size_t > global_work_size( 1, 128 ), local_work_size( 1, 32 );
  ciErrNum = clEnqueueNDRangeKernel( cqCommandQueue, ckSum, 1, NULL, & global_work_size[0], & local_work_size[ 0 ], 0, NULL, NULL );
  clCheckError( ciErrNum, "clEnqueuWriteBuffer d_b" );

  std::cout << "enquing d_c read" << std::endl;
  ciErrNum = clEnqueueReadBuffer( cqCommandQueue, d_c, CL_TRUE, 0, 128 * sizeof( float ), & h_c[ 0 ], 0, NULL, NULL );
  clCheckError( ciErrNum, "clEnqueuReadBuffer d_b" );

  // clFinish( cxGPUContext );

  bool any_bad = false;
  for ( int ii = 0; ii < 128; ++ii ) {
    if ( h_c[ ii ] != 16.f ) {
      any_bad = true;
      std::cout << "bad value " << ii << " " << h_c[ ii ] << " instead of 16" << std::endl;
    }
  }
  if ( ! any_bad ) {
    std::cout << "All values computed as expected!" << std::endl;
  }

  return 0;

}
