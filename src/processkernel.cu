#include <vector>
#include <string>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
//#include <omp.h>
#include <cmath>
#include "../include/processkernel.cuh"
#include "../include/l1ls.cuh"
#include "../include/pss.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "../lib/Eigen/Core"

using namespace Eigen;
using namespace std;

#define mMaxVelKmS	20

void handleError( cudaError_t err, const char *file, int line ) 
{
    if ( err != cudaSuccess ) 
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString(err), file, line );
        exit( EXIT_FAILURE );
    }
}
#define CHECK( err ) (handleError( err, __FILE__, __LINE__ ))
template <typename T>
__global__ void testKernel( const T *A,  const T *b, T *x, 
	T *u, T *f, T *newu, T *newx, T *newf, 
	T *dxu, T *z, T *tmpm, T *d1, T *d2, 
        T *gradphi, T *w, T *p,
	int m, int n, T lambda, T tar_gap, T eta, int pcgmaxi )
{
    
    solveL1LS<T>( A, b, x, u, f, newu, newx, newf, dxu, z, tmpm, d1, d2, 
		  gradphi, w, p, m, n, lambda, tar_gap, eta, pcgmaxi );
}



MatrixXd toeplitz( const VectorXd& wltmatrix, int ncol )
{
    const int window = wltmatrix.rows();
    
    MatrixXd W( window + ncol - 1, ncol );
    VectorXd w_temp(window + ncol - 1);
    
    for (int i = 0; i < ncol; i++)
    {
	w_temp = VectorXd::Constant(window + ncol - 1, 0.0);
	w_temp.segment(i, window) = wltmatrix;
	W.col(i) = w_temp;
    }

    return W;
}

void testL1LS()
{
    // Solve for A*x = b 
   // A =  1., 0.,  0.,  0.5,
   //      0., 1.,  0.2, 0.3,
   //      0., 0.1, 1.,  0.2
  // x0= 1, 0, 1 
  // b=A*x0
typedef double T;
    const int m = 3;
    const int n = 4;
    const int k = 2*n;
    const T lambda = 0.01;
    const T rel_tol = 0.01;

    std::cout << "True solution: (1, 0, 1, 0)  " << std::endl;
    T A[] = { 1., 0., 0., 0.5, 0., 1., 0.2, 0.3, 0., 0.1, 1., 0.2 };
    T b[] = { 1., 0.2, 1. };

    T x1[n];
    
    T dz[m], dtmpm[m]; // size m
    T du[n], dnewu[n], dnewx[n], dd1[n], dd2[n]; //size n
    T df[2*n], dnewf[2*n], ddxu[2*n], dgradphi[2*n], dw[2*n], dp[2*n];//size 2*n
    T tsec0 = Utilities::cpuSecond();
    solveL1LS<T>(A, b, x1, du, df, dnewu, dnewx, dnewf, ddxu, dz, dtmpm, 
	dd1, dd2, dgradphi, dw, dp, m, n, lambda, rel_tol, 1e-3, 2000);
    T cpulaps = Utilities::cpuSecond() - tsec0;
    std::cout << "CPU time:  " <<  cpulaps << "s. Sol:  ";
    for ( int i=0; i<n; ++i )
        std::cout << x1[i] << "\t";
    std::cout << endl;
    return 0;

    T x[n];
    T *d_A; //size mxn
    T *d_b, *d_z, *d_tmpm; // size m
    T *d_x, *d_u, *d_newu, *d_newx, *d_d1, *d_d2; //size n
    T *d_f, *d_newf, *d_dxu, *d_gradphi, *d_w, *d_p;//size 2*n

    CHECK( cudaMalloc( (void**)&d_A, m*n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_b, m*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_z, m*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_tmpm, m*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_x, n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_u, n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_newu, n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_newx, n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_d1, n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_d2, n*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_f, k*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_newf, k*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_dxu, k*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_gradphi, k*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_w, k*sizeof(T) ) );
    CHECK( cudaMalloc( (void**)&d_p, k*sizeof(T) ) );
    
    CHECK( cudaMemcpy(d_A, A, m*n*sizeof(T), cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_b, b, m*sizeof(T), cudaMemcpyHostToDevice) );
    
     tsec0 = Utilities::cpuSecond();
    testKernel<T><<<1,1>>>(d_A, d_b, d_x, d_u, d_f, d_newu, d_newx, d_newf, d_dxu,
	d_z, d_tmpm, d_d1, d_d2, d_gradphi, d_w, d_p, 
	m, n, lambda, rel_tol, 1e-3, 2000);
    cudaDeviceSynchronize();
    cpulaps = Utilities::cpuSecond() - tsec0;

    CHECK( cudaMemcpy(x, d_x, n*sizeof(T), cudaMemcpyDeviceToHost) );
	
    cudaFree( d_A ); 
    cudaFree( d_b ); 
    cudaFree( d_z ); 
    cudaFree( d_tmpm ); 
    cudaFree( d_x ); 
    cudaFree( d_u ); 
    cudaFree( d_newu ); 
    cudaFree( d_newx ); 
    cudaFree( d_d1 ); 
    cudaFree( d_d2 ); 
    cudaFree( d_f ); 
    cudaFree( d_newf ); 
    cudaFree( d_dxu ); 
    cudaFree( d_gradphi ); 
    cudaFree( d_w );  
    cudaFree( d_p );

    std::cout << "GPU time:  " <<  cpulaps << "s. Sol:  ";
    for ( int i=0; i<n; ++i )
        std::cout << x[i] << "\t";
    std::cout << endl;
}


void writeTrace( const IOPar::SetupInfo& su, const double *x, int sz, 
		 const std::vector<int>& nrsmps, const std::vector<int>& z0s )
{
    const int cnrz = sz/3;
    std::vector<float> vp(cnrz), vs(cnrz), den(cnrz), pi(cnrz);
    for ( size_t i=0; i<cnrz; ++i )
    {
        vp[i] = x[i];
        vs[i] = x[i+cnrz];
        den[i] = x[i+2*cnrz]; 
        pi[i] = (vp[i] - su.pic*vs[i])*den[i];
    }

    std::string vpnm( su.jobdir.c_str()); vpnm.append("/vp.o");
    std::string vsnm( su.jobdir.c_str()); vsnm.append("/vs.o");
    std::string dennm( su.jobdir.c_str()); dennm.append("/den.o");
    std::string pinm( su.jobdir.c_str()); pinm.append("/pi.o");
    Utilities::writeBinaryData( vpnm.c_str(), vp.data(), cnrz );
    Utilities::writeBinaryData( vsnm.c_str(), vs.data(), cnrz );
    Utilities::writeBinaryData( dennm.c_str(), den.data(), cnrz );
    Utilities::writeBinaryData( pinm.c_str(), pi.data(), cnrz );
}


bool process( const IOPar::SetupInfo& su, const std::vector<int>& h0,
	      const std::vector<int>& h1, std::string& err )
{
    int dev = 0;
    cudaDeviceProp devprop;
    CHECK(cudaGetDeviceProperties(&devprop,dev));
    printf("Using Device %d: %s\n", dev, devprop);
    CHECK(cudaSetDevice(dev));
    
    const int nrz = (su.t1-su.t0)/su.tstep+1;
    const int nrangles = (su.a1-su.a0)/su.astep+1;
    const int wvletsz = (su.sw[2]-su.sw[0])/su.sw[1]+1;

    const bool usehorconstrain = !su.horinp.empty(); 
    const size_t startidx = su.startcdp <= 0 ? 0 : su.startcdp-1;
    const size_t stopidx = startidx; //one trace test only for now
	//su.stopcdp <= 0 || su.stopcdp > su.nrcdp ? su.nrcdp-1 : su.stopcdp-1;

    MatrixXd singlewv;
    if ( su.wvtype==1 )
    {
	std::vector<float> wv;
	Utilities::fileBinaryRead<float>( wv, su.wvinp.c_str(), 0, 
					  nrangles*wvletsz );
	std::vector<double> wv1( wv.begin(), wv.end() );
	singlewv = Eigen::Map<MatrixXd>( wv1.data(), wvletsz, nrangles );
    }

    std::vector<int> nrsmps, z0s;
    size_t totalsmps = 0;
    for ( size_t idx=startidx; idx<=stopidx; idx++ )
    {
	const int smps = usehorconstrain ? (h1[idx]-h0[idx])/su.tstep+1 : nrz;
	totalsmps += smps;
	nrsmps.push_back( smps );
	z0s.push_back( (usehorconstrain ? (h0[idx]-su.t0)/su.tstep : 0) );
    }

    double *g_lambda = new (nothrow) double[stopidx-startidx+1];	
    double *g_angdata = new (nothrow) double[nrangles*totalsmps];
    double *g_wvformmat = new (nothrow) double[nrangles*totalsmps*totalsmps*3];
    if ( g_lambda==NULL || g_angdata==NULL || g_wvformmat==NULL )
    {
        delete[] g_wvformmat;
        delete[] g_angdata;
        delete[] g_lambda;
        std::cout << "Error: not enough memory to allocate data" << std::endl;
        return false;
    }

    for ( size_t idx=startidx; idx<=stopidx; idx++ )
    {
	const int cnrz = nrsmps[idx];
	const bool need_trim_data = cnrz < nrz;

	std::vector<float> ag;
	int sidx0 = idx*nrangles*nrz;
	int sz = nrangles*nrz;
	Utilities::fileBinaryRead<float>( ag, su.aginp.c_str(), sidx0, sz );
	MatrixXd s_data( cnrz, nrangles );
	if ( need_trim_data )
	{
	    for ( int i=0; i<nrangles; ++i )
	    {
		std::vector<double> ag1( ag.begin()+i*nrz+z0s[idx], 
			ag.begin()+i*nrz+z0s[idx]+cnrz );
		s_data.col(i) = Eigen::Map<VectorXd>(ag1.data(),cnrz);
	    }
	}
	else
	{
    	    std::vector<double> ag1( ag.begin(), ag.end() );
    	    s_data = Eigen::Map<MatrixXd>(ag1.data(),nrz,nrangles);
	}

	MatrixXd w_data;
	if ( su.wvtype==0 )
	{
    	    std::vector<float> wv;
    	    sidx0 = idx*nrangles*wvletsz;
    	    sz = nrangles*wvletsz;
    	    Utilities::fileBinaryRead<float>( wv, su.wvinp.c_str(), sidx0, sz );
    	    std::vector<double> wv1( wv.begin(), wv.end() );
    	    w_data = Eigen::Map<MatrixXd>( wv1.data(), wvletsz, nrangles );
	}
	else
	    w_data = singlewv;

	sidx0 = idx*nrz+z0s[idx];
	std::vector<float> bvp, bvs; 
	if ( !Utilities::fileBinaryRead<float>(bvp,su.vpinp.c_str(),sidx0,cnrz) ||
	     !Utilities::fileBinaryRead<float>(bvs,su.vsinp.c_str(),sidx0,cnrz) )
	{
	    cout << " Background model err at CDP " << idx+1 << endl;
	    return false;
	}
	
	VectorXd vsvpra(cnrz);
	for ( int i=0; i<cnrz; i++ )
	    vsvpra(i) = (double)bvs[i]/bvp[i];

	const int M = nrangles*cnrz ;
	const int N = cnrz*3;
	MatrixXd refmatrix = MatrixXd::Constant( M, N, 0.0 );
	MatrixXd waveformmatrix = MatrixXd::Zero( M, N );
	VectorXd angdata = VectorXd::Zero( M );
	size_t id0 = 0;
	for ( int i=startidx; i<idx; ++i )
	    id0 += nrsmps[i-startidx];
	//each cdp has nr of data:  id0*nrangles--id0*nrangles + nrangles*cnrz-1

	const double deg2rad = M_PI/180.0;
	for (int i=0; i<nrangles; ++i )
	{
	    const double angr = (su.a0+i*su.astep) * deg2rad;
	    const VectorXd& wlt = w_data.col(i);
	    double  normval = wlt.norm();
	    VectorXd tempwlt = wlt/normval;
	    const int nshift = floor( wlt.rows()*0.5 );

	    for ( int j=0; j<cnrz; ++j )
	    {
		refmatrix(cnrz*i+j,j) = 0.5 * ( 1 + pow(tan(angr), 2) );
		refmatrix(cnrz*i+j,j+cnrz) = -4 * pow(vsvpra(j)*sin(angr),2);
		refmatrix(cnrz*i+j,j+2*cnrz) = 0.5-2*pow(vsvpra(j)*sin(angr),2);
	    }

	    MatrixXd W_extend = toeplitz( tempwlt, cnrz );
	    MatrixXd W = W_extend.block( nshift, 0, cnrz, cnrz );
	    MatrixXd ref0 = refmatrix.block(cnrz*i, 0, cnrz, cnrz*3);
	    waveformmatrix.block(cnrz*i, 0, cnrz, cnrz*3) = W * ref0;
	    angdata.segment(cnrz*i, cnrz) = s_data.col(i) / normval;
	}

	const int ci = id0*nrangles;
        for ( int i=0; i<angdata.size(); ++i )
            g_angdata[ci+i] = angdata(i);

	VectorXd temp = 2*waveformmatrix.transpose()*angdata;
        g_lambda[idx-startidx] = 0.0001*temp.cwiseAbs().maxCoeff();
	//size for each cdp: nrangles*cnrz x cnrz*3;
	id0 = id0*id0*nrangles*3;
	for ( int r=0; r<M; ++r )
	    for ( int c=0; c<N; ++c )
		g_wvformmat[id0++] = waveformmatrix(r,c);
    }
    
    //Now, solve L1lS for all the CDPs
    const size_t m = nrangles*totalsmps;
    const size_t n = totalsmps*3;
    const size_t k = 2*n;
    const double rel_tol = 1e-4;

    //double *x = new (nothrow) double[n];
    double *x; x = (double *)malloc(n*sizeof(double));

    /* //CPU version test!
    double d_z[m], d_tmpm[m]; // size m
    double d_x[n], d_u[n], d_newu[n], d_newx[n], d_d1[n], d_d2[n]; //size n
    double d_f[2*n], d_newf[2*n], d_dxu[2*n], d_gradphi[2*n], d_w[2*n], d_p[2*n];//size 2*n
    solveL1LS<double>(g_wvformmat, g_angdata, d_x, d_u, d_f, d_newu, d_newx, d_newf, d_dxu,
	d_z, d_tmpm, d_d1, d_d2, d_gradphi, d_w, d_p, 
	m, n, g_lambda[0], rel_tol, 1e-3, 2000); */

    double *d_A; //size mxn
    double *d_b, *d_z, *d_tmpm; // size m
    double *d_x, *d_u, *d_newu, *d_newx, *d_d1, *d_d2; //size n
    double *d_f, *d_newf, *d_dxu, *d_gradphi, *d_w, *d_p;//size 2*n
    CHECK( cudaMalloc( &d_A, m*n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_b, m*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_z, m*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_tmpm, m*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_x, n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_u, n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_newu, n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_newx, n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_d1, n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_d2, n*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_f, k*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_newf, k*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_dxu, k*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_gradphi, k*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_w, k*sizeof(double) ) );
    CHECK( cudaMalloc( (void**)&d_p, k*sizeof(double) ) );
    
    CHECK( cudaMemcpy(d_A, g_wvformmat, m*n*sizeof(double), cudaMemcpyHostToDevice) );
    CHECK( cudaMemcpy(d_b, g_angdata, m*sizeof(double), cudaMemcpyHostToDevice) );
    cudaMemset( (void*)d_x, 0.0, n*sizeof(double) );
    cudaMemset( (void*)d_u, 1.0, n*sizeof(double) );
    cudaMemset( (void*)d_f, -1.0, k*sizeof(double) );
    cudaMemset( (void*)d_dxu, 0.0, k*sizeof(double) );

    double tsec0 = Utilities::cpuSecond();
    testKernel<double><<<1,1>>>(d_A, d_b, d_x, d_u, d_f, d_newu, d_newx, d_newf, d_dxu,
	d_z, d_tmpm, d_d1, d_d2, d_gradphi, d_w, d_p, 
	m, n, g_lambda[0], rel_tol, 1e-3, 2000);
    cudaDeviceSynchronize();
    double tlaps = Utilities::cpuSecond() - tsec0;
    cout << "Time cost: " << tlaps << " s" << endl;

    CHECK( cudaMemcpy(x, d_x, n*sizeof(double), cudaMemcpyDeviceToHost) );
	
    cudaFree( d_A ); 
    cudaFree( d_b ); 
    cudaFree( d_z ); 
    cudaFree( d_tmpm ); 
    cudaFree( d_x ); 
    cudaFree( d_u ); 
    cudaFree( d_newu ); 
    cudaFree( d_newx ); 
    cudaFree( d_d1 ); 
    cudaFree( d_d2 ); 
    cudaFree( d_f ); 
    cudaFree( d_newf ); 
    cudaFree( d_dxu ); 
    cudaFree( d_gradphi ); 
    cudaFree( d_w );  
    cudaFree( d_p );

    writeTrace( su, x, n, nrsmps, z0s );
    cout << "Process finished! " << endl;

    delete[] x;
    return 0;
}


