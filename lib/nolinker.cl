#include <pyopencl-ranluxcl.cl>
#include <cl/gpu_mat_vec.cl>

#define GI get_global_id(1)
#define LI get_local_id(0)
#define ONLY1 if(get_local_id(0)==0)

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

bool need_to_check_psi(struct VEC const v) {
	if( v.z > 46.9102 ) return false;
	if( v.z < 9.0 ) return true;
	if( dist2v(v,vec(46.67326747,50.46076527,21.15043826)) > 2649.58996563299) return false;
	return true;
}

inline bool clash_check_psi(
	struct VEC const v,
	__constant struct VEC const *xyz,
	__constant ushort2 const *grid
){
	float const x = v.x;
	float const y = v.y;
	float const z = v.z;
	bool clash = false;
	int const ix	= (x<0) ? 0 : min(nxpsi-1,(int)(x/grid_size));
	int const iy0	= (y<0) ? 0 : y/grid_size;
	int const iz0	= (z<0) ? 0 : z/grid_size;
	int const iyl = max(0,iy0-1);
	int const izl = max(0,iz0-1);
	int const iyu = min((int)nypsi,iy0+2);
	int const izu = min((int)nzpsi,iz0+2);
	for(int iy = iyl; iy < iyu; ++iy) {
		for(int iz = izl; iz < izu; ++iz) {
			int const ig = ix+nxpsi*iy+nxpsi*nypsi*iz;
			int const igl = grid[ig].x;
			int const igu = grid[ig].y;
			for(int i = igl; i < igu; ++i) {
				struct VEC const a2 = xyz[i];
				float const d2 = (x-a2.x)*(x-a2.x) + (y-a2.y)*(y-a2.y) + (z-a2.z)*(z-a2.z);
				clash |= (d2 < 16.0);
			}
		}
	}
	return clash;
}

inline bool clash_check_hyd(
	struct VEC const v,
	__constant struct VEC const *xyz,
	__constant ushort2 const *grid
){
	float const x = v.x;
	float const y = v.y;
	float const z = v.z;
	bool clash = false;
	int const ix	= (x<0) ? 0 : min(nxhyd-1,(int)(x/grid_size));
	int const iy0	= (y<0) ? 0 : y/grid_size;
	int const iz0	= (z<0) ? 0 : z/grid_size;
	int const iyl = max(0,iy0-1);
	int const izl = max(0,iz0-1);
	int const iyu = min((int)nyhyd,iy0+2);
	int const izu = min((int)nzhyd,iz0+2);
	for(int iy = iyl; iy < iyu; ++iy) {
		for(int iz = izl; iz < izu; ++iz) {
			int const ig = ix+nxhyd*iy+nxhyd*nyhyd*iz;
			int const igl = grid[ig].x;
			int const igu = grid[ig].y;
			for(int i = igl; i < igu; ++i) {
				struct VEC const a2 = xyz[i];
				float const d2 = (x-a2.x)*(x-a2.x) + (y-a2.y)*(y-a2.y) + (z-a2.z)*(z-a2.z);
				clash |= (d2 < 16.0);
			}
		}
	}
	return clash;
}

__kernel void my_test_kernel(
	__constant struct VEC const *psi, __constant ushort2 const *psigrid,
	__constant struct VEC const *hyd, __constant ushort2 const *hydgrid,
	__constant struct VEC const *pet, __constant ushort2 const *petgrid,
	__global   struct VEC const *init_lkh,
	__global   struct VEC const *init_lkp,
	__global ranluxcl_state_t *ranluxcltab,
	__global uint *python_seed,
	__global struct VEC   *out,
	__global struct XFORM *xout,
	__global float *rout,
	__global struct VEC *vout,
	__global int *status_global,
	__global int   *stat,
	__global float *fstat,
	__global uint *hist,
	__global uint *hist6
){
	__local struct VEC psi_et,hyd_et,pet_et,hyd_cen,pet_cen;
	__local struct XFORM xh,xp,xhr,move;
	__local bool p_or_h,mc_fail;
	__local float mc_rand,minz[LS],boltz,score,last_score;
	__local int clash,nlnk,idx,residx,failrun,status,samp_delay;
	ONLY1 psi_et = vec(52.126627, 53.095875, 23.992003);
	ONLY1 status = status_global[2*GI+0];
	ONLY1 samp_delay= status_global[2*GI+1];
	
	if(status) { // continuing
		ONLY1 {
			xh = xout[3*GI+0];
		    xp = xout[3*GI+1];
		}
	} else { // initialize
		ONLY1 { 
			myrand_init(GI+*python_seed,ranluxcltab);
			xh = multxx( stub(init_lkh[90],init_lkh[91],init_lkh[92]) , hyd_stub ); 			    
			xp = multxx( stub(init_lkp[90],init_lkp[91],init_lkp[92]) , pet_stub );			
			samp_delay = 0; // < 0 -> OK >= 0 not "equilibrated" yet
		}
	}
    barrier(CLK_LOCAL_MEM_FENCE);

	for(int ITER = 0; ITER < NGPUITER; ++ITER) { // main loop
		ONLY1 { // compute next move 
	    	stat[nstat*GI+1] += ((status>100) ? 1 : 0);
			samp_delay = (samp_delay > 100) ? -1 : samp_delay; // start recording
			samp_delay = (   status < -100) ?  1 : samp_delay; // start delay count
			samp_delay = (   status >  100) ?  0 : samp_delay; // null
			status = (status < -100) ?  1 : status; // if 1000 nosample accepts, start sampling
			status = (status >  100) ? -1 : status; // if 1000 sample fails in a row, stop sampling
			float4 rand = myrand(ranluxcltab);
			p_or_h = rand.x < 0.5; 
			mc_rand = rand.x*2.0; // discard used 2
			if(mc_rand > 1.0) mc_rand -= 1.0;
			float4 randr = myrand(ranluxcltab);
			float4 randc = myrand(ranluxcltab);
			struct VEC trn = vec((randc.x-0.5)/1.0,(randc.y-0.5)/1.0,(randc.z-0.5)/1.0);
			struct VEC axs = vec(randr.x,randr.y,randr.z );
			pet_cen = multxv(xp,vec(12.6810557 ,15.45869573,14.22168096));
			hyd_cen = multxv(xh,vec(21.09134627,36.96850701,23.76050228));
			struct VEC cen = (p_or_h) ? pet_cen : hyd_cen;
			struct MAT R = rotation_matrix( axs, (randr.w-0.5)/1.0 );
			move = xform( R, addvv( subvv(cen,multmv(R,cen)) , trn ) );
//			move = xform( R,        subvv(cen,multmv(R,cen))         );			
			if(p_or_h) { xp = multxx(move,xp); } else { xh = multxx(move,xh); xhr = xrev(xh); }
			pet_et  = multxv(xp,vec( 5.468269  ,20.041713  ,11.962524  ));
			pet_cen = multxv(xp,vec(12.6810557 ,15.45869573,14.22168096));
			hyd_et  = multxv(xh,vec(11.842096  ,39.050866  ,19.269006  ));
			hyd_cen = multxv(xh,vec(21.09134627,36.96850701,23.76050228));
			clash = false;
		}

		ONLY1 score = native_sqrt(dist2v(hyd_et,pet_et)); // monte-carlo
		ONLY1 mc_fail = ((status>0) ? ( mc_rand > native_exp((last_score-score)/temperature) ) : false );
		if(mc_fail) {
			ONLY1 {
				struct XFORM undo = xrev(move);
				ONLY1 if(p_or_h) xp = multxx(undo,xp); else xh = multxx(undo,xh);							
				if(status > 0) stat[nstat*GI+3+(p_or_h?1:0)] += 1;
				ONLY1 status += ((status > 0) ? 1 : 0); // add a fail to status if sampling
		    	ONLY1 stat[nstat*GI+0] += (samp_delay<0) ? 1 : 0;
			}
			continue;
		}

		{ // clash check 
			int rclash = false;			
			minz[LI] = 9e9f;
			if(p_or_h) {
				for(size_t i = 0; i < npet; i+=LS) minz[LI] = min( minz[LI], (i+LI<npet) ? multxv(xp,pet[i+LI]).z : 9e9f );
				for(uint c=LS/2;c>0;c/=2) { barrier(CLK_LOCAL_MEM_FENCE); if(c>LI) minz[LI] = min(minz[LI],minz[LI+c]); }
				barrier(CLK_LOCAL_MEM_FENCE);
			  	if(minz[0] < -3.0) { ONLY1 clash=true; goto DONE_CLASH_CHECK; }
			
				// linkp && linkh vs pet
				for(size_t i = 0; i < npet; i+=LS) {
					if( i+LI >= npet ) break;
					struct VEC const v = multxv(xp,pet[i+LI]);
					if(need_to_check_psi(v)) rclash |= clash_check_psi(v,psi,psigrid);						
				}		
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
			} else {
				for(size_t i = 0; i < nhyd; i+=LS) minz[LI] = min( minz[LI], (i+LI<nhyd) ? multxv(xh,hyd[i+LI]).z : 9e9f );
				for(uint c=LS/2;c>0;c/=2) { barrier(CLK_LOCAL_MEM_FENCE); if(c>LI) minz[LI] = min(minz[LI],minz[LI+c]); }
				barrier(CLK_LOCAL_MEM_FENCE);
				if(minz[0] < -3.0) { ONLY1 clash=true; goto DONE_CLASH_CHECK; }

				for(size_t i = 0; i < nhyd; i+=LS) {
					if( i+LI >= nhyd ) break;
					struct VEC const v = multxv(xh,hyd[i+LI]);
					if(need_to_check_psi(v)) rclash |= clash_check_psi(v,psi,psigrid);			
				}				
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
			}			
			if( dist2v(hyd_cen,pet_cen) < 4057.24684269 ) {
				struct XFORM const xf = multxx(xhr,xp);
				// xout[3*GI+2] = xf;
				for(size_t i = 0; i < npet; i+=LS) {
					if( i+LI >= npet ) break;
					rclash |= clash_check_hyd(multxv(xf,pet[i+LI]),hyd,hydgrid);
				}				
			}
			DONE_CLASH_CHECK: ;
			if(rclash) clash=true;
		}
		
		if( clash ) { // undo move if not accepted 
			struct XFORM undo = xrev(move);
			ONLY1 if(p_or_h) xp = multxx(undo,xp); else xh = multxx(undo,xh);			
			ONLY1 status += ( (status > 0) ? 1 : 0 ); // add a fail to status if sampling
		} else ONLY1 { // ACCEPT!
			last_score = score;
			vout[2*GI+0] = hyd_et;
			vout[2*GI+1] = pet_et;
			status = ( (status > 0) ? 1 : status-1 ); // if sampling, reset to 1, else sub 1
			samp_delay = (samp_delay > 0) ? samp_delay+1 : samp_delay; // incr iff counting
			if( !clash && status>=0 && samp_delay<0 ) {
		   		stat[nstat*GI+2]++;
				hist[min(299u,((uint)score))]++;
				pet_et  = multxv(xp,vec( 5.468269  ,20.041713  ,11.962524  ));
				hyd_et  = multxv(xh,vec(11.842096  ,39.050866  ,19.269006  ));					
				struct VEC d = multmv( xhr.R, subvv(hyd_et,pet_et) );
				if( length2v(d) <= 256.0) {
					int ix = max(0,min( 8,(int)((d.x- 7.0)/1.0)));
					int iy = max(0,min(19,(int)((d.y+12.0)/1.0)));
					int iz = max(0,min(22,(int)((d.z+ 9.0)/1.0)));
					struct MAT m = multmm(xhr.R,xp.R);
					float theta = -asin(m.zx);
					float costheta = native_cos(theta);
					float psi   = atan2(m.zy/costheta,m.zz/costheta);
					float phi   = atan2(m.yx/costheta,m.xx/costheta);
					theta *= 57.29578;
					psi   *= 57.29578;
					phi   *= 57.29578;
					theta +=  90.0;
					psi   += 180.0;
					phi   += (phi<0) ? 270.0 : -90.0;
					int itheta = max(0,min(10,(int)(theta/15.0)-1));
					int ipsi   = max(0,min(23,(int)(  psi/15.0)));
					int iphi   = max(0,min(11,(int)(  phi/15.0)));
					hist6[ ix*20*23*11*24*12 + iy*23*11*24*12 + iz*11*24*12 + itheta*24*12 + ipsi*12 + iphi ]++;
				}//        9,19,23,11,24,12
			}
		}
		
		ONLY1 { // stats 
			if(status>0) {
				stat[nstat*GI+3+(p_or_h?1:0)] += 1;
	    		stat[nstat*GI+5+(p_or_h?1:0)] += (clash) ? 0 : 1;
			}
	    	ONLY1 stat[nstat*GI+0] += (samp_delay<0) ? 1 : 0;
		}

	}		
	
	// OUTPUT
//	barrier(CLK_LOCAL_MEM_FENCE);
	ONLY1 xout[3*GI+0] = xh;
	ONLY1 xout[3*GI+1] = xp;
	
	ONLY1 status_global[2*GI+0] = status;
	ONLY1 status_global[2*GI+1] = samp_delay;
	
}






















