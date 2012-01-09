
#include <cl/gpu_mat_vec.cl>

#define GI get_global_id(1)
#define LI get_local_id(0)
#define LS 64

int clash_check_xyzhash(
	float const x, float const y, float const z,
	__constant struct VEC const *xyz,
	__constant ushort2 const *grid
){
	int count = 0;
    int const ix   = (x<0) ? 0 : min(%(xsize)i-1,(int)(x/%(grid_size)i));
    int const iy0  = (y<0) ? 0 : y/%(grid_size)i;
    int const iz0  = (z<0) ? 0 : z/%(grid_size)i;
    int const iyl = max(0,iy0-1);
    int const izl = max(0,iz0-1);
    int const iyu = min((int)%(ysize)i,iy0+2);
    int const izu = min((int)%(zsize)i,(int)iz0+2);
    for(int iy = iyl; iy < iyu; ++iy) {
      for(int iz = izl; iz < izu; ++iz) {
        int const ig = ix+%(xsize)i*iy+%(xsize)i*%(ysize)i*iz;
        int const igl = grid[ig].x;
        int const igu = grid[ig].y;
        for(int i = igl; i < igu; ++i) {
          struct VEC const a2 = xyz[i];
          float const d2 = (x-a2.x)*(x-a2.x) + (y-a2.y)*(y-a2.y) + (z-a2.z)*(z-a2.z);
          if( d2 < 16.0 ) {
            ++count;
          }
        }
      }
    }
	return count;
}


__kernel void test_xyzhash(	
	__constant struct VEC const *xyz,
	__constant ushort2 const *grid,
	__global int *out
){
	__local int count[LS];
	count[LI] = 0;
	
	float grid_size2 = %(grid_size)i*%(grid_size)i;
	
	float z = (float)GI;
	float y = (float)LI;
	for(float x = 0.0; x < 64.0; x += 1.0) {
		count[LI] += clash_check_xyzhash(x,y,z,xyz,grid);
			
		for(int i = 0; i < %(nxyz)i-5; i+=5) {
			count[LI] -= (dist2v(xyz[i+0],vec(x,y,z)) < 16.0) ? 1 : 0;
			count[LI] -= (dist2v(xyz[i+1],vec(x,y,z)) < 16.0) ? 1 : 0;
			count[LI] -= (dist2v(xyz[i+2],vec(x,y,z)) < 16.0) ? 1 : 0;
			count[LI] -= (dist2v(xyz[i+3],vec(x,y,z)) < 16.0) ? 1 : 0;
			count[LI] -= (dist2v(xyz[i+4],vec(x,y,z)) < 16.0) ? 1 : 0;
		}

	}
			
	for(uint c=LS/2;c>0;c/=2) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if(c>LI) count[LI] += count[LI+c];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(LI==0) atom_add(out,count[0]);
}
