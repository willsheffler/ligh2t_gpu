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
	__global uint *hist
){
	if(nlkh < 9 || nlkh < 9) return;
	__local struct VEC lkh[nlkh],lkp[nlkp],psi_et,hyd_et,pet_et,hyd_cen,pet_cen;
	__local struct XFORM xh,xp,xhr,move;
	__local bool p_or_h,mc_fail;
	__local float mc_rand,minz[LS],boltz,score,last_score;
	__local int clash,nlnk,idx,residx,failrun,status,samp_delay;
	ONLY1 psi_et = vec(52.126627, 53.095875, 23.992003);
	ONLY1 status = status_global[2*GI+0];
	ONLY1 samp_delay= status_global[2*GI+1];
	
	if(status) { // continuing
		for(size_t i=0; i<nlkh; i+=LS) if((i+LI)<nlkh) lkh[i+LI] = out[GI*ntot     +i+LI];
		for(size_t i=0; i<nlkp; i+=LS) if((i+LI)<nlkp) lkp[i+LI] = out[GI*ntot+nlkh+i+LI];
		ONLY1 {
			xh = xout[3*GI+0];
		    xp = xout[3*GI+1];
		}
	} else { // initialize
		for(size_t i = 0; i < nlkh; i+=LS) if((i+LI)<nlkh) lkh[i+LI]=init_lkh[i+LI];
		for(size_t i = 0; i < nlkp; i+=LS) if((i+LI)<nlkp) lkp[i+LI]=init_lkp[i+LI];
		ONLY1 { 
			myrand_init(GI+*python_seed,ranluxcltab);
		    xh = multxx( stub(lkh[nlkh-3],lkh[nlkh-2],lkh[nlkh-1]) , hyd_stub );
		    xp = multxx( stub(lkp[nlkp-3],lkp[nlkp-2],lkp[nlkp-1]) , pet_stub ); 
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
			residx = (uint)(rand.y*(float)(p_or_h?nlkp/3:nlkh/3)-2.0) + 1;
			bool phipsi = rand.z < 0.5;
			float ang = (2.0*rand.w-1.0) * ((status > 0) ? 0.1 : 3.14159); // big move if not sampling
			idx = 3*residx + (phipsi?0:1);
			nlnk = p_or_h?nlkp:nlkh; 
			struct VEC cen = (p_or_h?lkp:lkh)[idx];
			struct VEC axs = subvv( (p_or_h?lkp:lkh)[idx+1], (p_or_h?lkp:lkh)[idx] );
			struct MAT R = rotation_matrix( axs, ang );
			move = xform( R, subvv(cen,multmv(R,cen)) );
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
				if(status > 0) stat[nstat*GI+2+(p_or_h?1:0)] += 1;
				ONLY1 status += ((status > 0) ? 1 : 0); // add a fail to status if sampling
		    	ONLY1 stat[nstat*GI+0] += (samp_delay<0) ? 1 : 0;
			}
			continue;
		}

		for(int i = idx+2; i < nlnk; i+=LS) { // xform linker coords 
			if(LI+i < nlnk) (p_or_h?lkp:lkh)[LI+i] = multxv(move,(p_or_h?lkp:lkh)[LI+i]);
		}

		{ // clash check 
			int rclash = false;			
			minz[LI] = 9e9f;
			if(p_or_h) {
			    ONLY1 xp = multxx( stub(lkp[nlkp-3],lkp[nlkp-2],lkp[nlkp-1]) , pet_stub ); 			    
				for(size_t i = 0; i < nlkp; i+=LS) if(i+LI < nlkp && lkp[i+LI].z < -3.0) rclash = true;
				for(size_t i = 0; i < npet; i+=LS) minz[LI] = min( minz[LI], (i+LI<npet) ? multxv(xp,pet[i+LI]).z : 9e9f );
				for(uint c=LS/2;c>0;c/=2) { barrier(CLK_LOCAL_MEM_FENCE); if(c>LI) minz[LI] = min(minz[LI],minz[LI+c]); }
				barrier(CLK_LOCAL_MEM_FENCE);
			  	if(minz[0] < -3.0) { ONLY1 clash=true; goto DONE_CLASH_CHECK; }
			
				for(size_t ip = 0; ip < nlkp; ip+=LS) {
					if(ip+LI >= nlkp) break;
					struct VEC const vp = lkp[ip+LI];
					for(size_t ih  =        0; ih < nlkh-5; ih+=5) {
						rclash |= ( dist2v(vp,lkh[ih+0]) < 16.0 );
						rclash |= ( dist2v(vp,lkh[ih+1]) < 16.0 );
						rclash |= ( dist2v(vp,lkh[ih+2]) < 16.0 );
						rclash |= ( dist2v(vp,lkh[ih+3]) < 16.0 );
						rclash |= ( dist2v(vp,lkh[ih+4]) < 16.0 );
					}
					for(size_t ip2 = ip+LI+4; ip2 < nlkp-5; ip2+=5) {
						rclash |= ( dist2v(vp,lkp[ip2+0]) < 16.0 );
						rclash |= ( dist2v(vp,lkp[ip2+1]) < 16.0 );
						rclash |= ( dist2v(vp,lkp[ip2+2]) < 16.0 );
						rclash |= ( dist2v(vp,lkp[ip2+3]) < 16.0 );
						rclash |= ( dist2v(vp,lkp[ip2+4]) < 16.0 );
					}
				}
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
				// linkp vs psi
				for(size_t ilp = 6; ilp < nlkp; ilp+=LS) {       // skip start of linkp vs. psi
					rclash |= clash_check_psi( (ilp+LI<nlkp) ? lkp[ilp+LI] : vec(9e9,9e9,9e9) ,psi,psigrid);
				}		
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
				// linkp vs hyd
				for(size_t i = 0; i < nlkp; i+=LS) {       // skip start of linkp vs. psi
					if(i+LI >= nlkp) break;
					rclash |= clash_check_hyd( multxv(xhr,lkp[i+LI]) ,hyd,hydgrid);
				}		
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
				// linkp && linkh vs pet
				for(size_t i = 0; i < npet; i+=LS) {
					if( i+LI >= npet ) break;
					struct VEC const v = multxv(xp,pet[i+LI]);
					for(size_t ilp = 0; ilp < nlkp-11; ilp+=5) {      // skip end of linkp vs pet
						rclash |= ( dist2v(v,lkp[ilp+0]) < 16.0 );
						rclash |= ( dist2v(v,lkp[ilp+1]) < 16.0 );
						rclash |= ( dist2v(v,lkp[ilp+2]) < 16.0 );								
						rclash |= ( dist2v(v,lkp[ilp+3]) < 16.0 );
						rclash |= ( dist2v(v,lkp[ilp+4]) < 16.0 );								
					}
					for(size_t ilh = 0; ilh < nlkh-5; ilh+=5) {
						rclash |= ( dist2v(v,lkh[ilh+0]) < 16.0 );
						rclash |= ( dist2v(v,lkh[ilh+1]) < 16.0 );
						rclash |= ( dist2v(v,lkh[ilh+2]) < 16.0 );								
						rclash |= ( dist2v(v,lkh[ilh+3]) < 16.0 );
						rclash |= ( dist2v(v,lkh[ilh+4]) < 16.0 );								
					}
					if(need_to_check_psi(v)) rclash |= clash_check_psi(v,psi,psigrid);						
				}		
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
			} else {
				ONLY1 xh = multxx( stub(lkh[nlkh-3],lkh[nlkh-2],lkh[nlkh-1]) , hyd_stub );				
				for(size_t i = 0; i < nlkh; i+=LS) if(i+LI < nlkh && lkh[i+LI].z < -3.0) rclash = true;
				for(size_t i = 0; i < nhyd; i+=LS) minz[LI] = min( minz[LI], (i+LI<nhyd) ? multxv(xh,hyd[i+LI]).z : 9e9f );
				for(uint c=LS/2;c>0;c/=2) { barrier(CLK_LOCAL_MEM_FENCE); if(c>LI) minz[LI] = min(minz[LI],minz[LI+c]); }
				barrier(CLK_LOCAL_MEM_FENCE);
				if(minz[0] < -3.0) { ONLY1 clash=true; goto DONE_CLASH_CHECK; }

				for(size_t ih = 0; ih < nlkh; ih+=LS) {
					if(ih+LI >= nlkh) break;
					struct VEC const vh = lkh[ih+LI];
					for(size_t ip  =       0; ip  < nlkp-5; ip+=5 ) {
						rclash |= ( dist2v(vh,lkp[ip+0]) < 16.0 );
						rclash |= ( dist2v(vh,lkp[ip+1]) < 16.0 );
						rclash |= ( dist2v(vh,lkp[ip+2]) < 16.0 );
						rclash |= ( dist2v(vh,lkp[ip+3]) < 16.0 );
						rclash |= ( dist2v(vh,lkp[ip+4]) < 16.0 );
					}
					for(size_t ih2 = ih+LI+4; ih2 < nlkh-5; ih2+=5) {
						rclash |= ( dist2v(vh,lkh[ih2+0]) < 16.0 );
						rclash |= ( dist2v(vh,lkh[ih2+1]) < 16.0 );
						rclash |= ( dist2v(vh,lkh[ih2+2]) < 16.0 );
						rclash |= ( dist2v(vh,lkh[ih2+3]) < 16.0 );
						rclash |= ( dist2v(vh,lkh[ih2+4]) < 16.0 );
					}
				}
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
				for(size_t ilh = 6; ilh < nlkh; ilh+=LS) {       // skip start of linkp vs. psi
					rclash |= clash_check_psi( (ilh+LI<nlkh) ? lkh[ilh+LI] : vec(9e9,9e9,9e9) ,psi,psigrid);
				}						
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
				// hyd vs. hlink & plink
				for(size_t i = 0; i < nlkh-9; i+=LS) {        // skip end of linkh vs hyd
					if( i+LI >= nlkh-9 ) break;
					rclash |= clash_check_hyd( multxv(xhr,lkh[i+LI]), hyd, hydgrid );					
				}
				for(size_t i = 0; i < nlkp; i+=LS) {
					if( i+LI >= nlkp ) break;
					rclash |= clash_check_hyd( multxv(xhr,lkp[i+LI]) ,hyd, hydgrid );					
				}

				for(size_t i = 0; i < nhyd; i+=LS) {
					if( i+LI >= nhyd ) break;
					struct VEC const v = multxv(xh,hyd[i+LI]);
					if(need_to_check_psi(v)) rclash |= clash_check_psi(v,psi,psigrid);			
				}				
				if(rclash) clash=true; if(clash) goto DONE_CLASH_CHECK;
				for(size_t i = 0; i < npet; i+=LS) {
					if( i+LI >= npet ) break;
					struct VEC const v = multxv(xp,pet[i+LI]);
					for(size_t ilh = 0; ilh < nlkh-5; ilh+=5) {
						rclash |= ( dist2v(v,lkh[ilh+0]) < 16.0 );
						rclash |= ( dist2v(v,lkh[ilh+1]) < 16.0 );
						rclash |= ( dist2v(v,lkh[ilh+2]) < 16.0 );								
						rclash |= ( dist2v(v,lkh[ilh+3]) < 16.0 );
						rclash |= ( dist2v(v,lkh[ilh+4]) < 16.0 );								
					}
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
			for(int i = idx+2; i < nlnk; i+=LS) 
				if(LI+i < nlnk) (p_or_h?lkp:lkh)[LI+i] = multxv(undo,(p_or_h?lkp:lkh)[LI+i]);				
			ONLY1 if(p_or_h) xp = multxx(undo,xp); else xh = multxx(undo,xh);			
			ONLY1 status += ( (status > 0) ? 1 : 0 ); // add a fail to status if sampling
		} else { // ACCEPT!
			ONLY1 last_score = score;
			ONLY1 vout[2*GI+0] = hyd_et;
			ONLY1 vout[2*GI+1] = pet_et;
			ONLY1 status = ( (status > 0) ? 1 : status-1 ); // if sampling, reset to 1, else sub 1
			ONLY1 hist[300u*GI+min(299u,((uint)score))] += ( (clash || status < 0 || samp_delay >= 0) ? 0 : 1 );
			samp_delay = (samp_delay > 0) ? samp_delay+1 : samp_delay; // incr iff counting
		}
		
		ONLY1 { // stats 
			if(status>0) {
				stat[nstat*GI+2+(p_or_h?1:0)] += 1;
	    		stat[nstat*GI+4+(p_or_h?1:0)] += (clash) ? 0 : 1;
			}
	    	stat[nstat*GI+0] += (samp_delay<0) ? 1 : 0;
		}

	}		
	
	// OUTPUT
//	barrier(CLK_LOCAL_MEM_FENCE);
	for(size_t i=0; i<nlkh; i+=LS) if((i+LI)<nlkh) out[GI*ntot     +i+LI] = lkh[i+LI];
	for(size_t i=0; i<nlkp; i+=LS) if((i+LI)<nlkp) out[GI*ntot+nlkh+i+LI] = lkp[i+LI];
	ONLY1 xout[3*GI+0] = xh;
	ONLY1 xout[3*GI+1] = xp;
	
	ONLY1 status_global[2*GI+0] = status;
	ONLY1 status_global[2*GI+1] = samp_delay;
	
}






















