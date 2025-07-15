#ifndef _CUDA_L1LS_H_
#define _CUDA_L1LS_H_

//#include <fstream>
//#include <iostream>


#define mAbs(x) ((x) > 0. ? x : -(x))
#define mMax(x,y)       ((x)>(y) ? (x) : (y))
#define mMin(x,y)       ((x)<(y) ? (x) : (y))

#define mDBL_MAX	1e+28
#define MU		2	//updating parameter of t 
#define MAX_NT_ITER	400	//IPM max Newton iteration
#define ALPHA		0.01	//minimum fraction of decrease in the objective
#define BETA		0.5	//stepsize decrease factor
#define MAX_LS_ITER	100	//maximum backtracking line search iteration

//s1*(A*x)+s2*y or s1*At*x+s2*y, A is a mxn matrix, 
//s1, s2 are scales, x, y are vectors: A(i,j) = A[i*n+j] //rowwise store
template <typename T>
__host__ __device__ void AxPy( const T *A, const T *x,
	 const T *y, T *res, T s1, T s2, int m, int n, bool transpose )
{
    T ax;
    if ( transpose )
    {
	if ( y==NULL || s2==0.0 )
	{
	    for ( int i=0; i<n; ++i )
	    {
		ax = 0.;
		for ( int j=0; j<m; ++j )
		    ax += A[j*n+i] * x[j];
		res[i] = s1 * ax;
	    }
	}
	else
	{
	    for ( int i=0; i<n; ++i )
	    {
		ax = 0.;
		for ( int j=0; j<m; ++j )
		    ax += A[j*n+i] * x[j];

		res[i] = s2 * y[i] + s1 * ax;
	    }
	}
    }
    else
    {
	if ( y==NULL || s2==0.0 )
	{
	    for ( int i=0; i<m; ++i )
	    {
		ax = 0.;
		for ( int j=0; j<n; ++j )
		    ax += A[i*n+j] * x[j];

		res[i] = s1 * ax;
	    }
	}
	else
	{
	    for ( int i=0; i<m; ++i )
	    {
		ax = 0.;
		for ( int j=0; j<n; ++j )
		    ax += A[i*n+j] * x[j];

		res[i] = s2 * y[i] + s1 * ax;
	    }
	}
    }
}

template <typename T>
__host__ __device__ T Norm2( const T *v, int sz )
{
    T sum = 0.0;
    for ( int i=0; i<sz; ++i )
	sum += (v[i]*v[i]);
    return sqrt(sum);
}

template <typename T>
__host__ __device__ T Norm1( const T *v, int sz )
{
    T sum = 0.0;
    for ( int i=0; i<sz; ++i )
	sum += mAbs(v[i]);
    return sum;
}

template <typename T>
__host__ __device__ T NormInf( const T *v, int sz )
{
    T max = 0.0;
    T absval;
    for ( int i=0; i<sz; ++i )
    {
	absval = mAbs(v[i]);
	if ( max < absval )
	    max = absval;
    }

    return max;
}


template <typename T>
__host__ __device__ T Xdot( const T *x, const T *y, int sz ) 
{
    T sum = 0.;
    for ( int i=0; i<sz; ++i )
	sum += x[i]*y[i];
    return sum;
}

template <typename T> //perform a*X+b*Y
__host__ __device__ void LinearVec( T *x, T *y, T *res, int sz, T a, T b ) 
{
    for ( int i=0; i<sz; ++i )
	res[i] = a*x[i] + b*y[i];
}

template <typename T>
__host__ __device__ void pcgAX( const T* x, //2n 
		       const T* A, //mxn 
		       const T* d1, //n 
		       const T* d2, //n
		       T* res, //2n
       		       int m, int n,
		      T *tmpn,
		      T *tmpm )
{
    int i;
    for ( i=0; i<n; ++i )
	tmpn[i] = x[i];

    AxPy<T>( A, tmpn, NULL, tmpm, 1,0, m, n, false );
    AxPy<T>( A, tmpm, NULL, tmpn, 2,0, m, n, true );

    for ( i=0; i<n; ++i )
    {
	res[i] = d1[i]*x[i]+d2[i]*x[i+n] + tmpn[i]; 
	res[i+n] = d1[i]*x[i+n] + d2[i]*x[i];
    }
}


template <typename T>
__host__ __device__ void pcgSolver(const T* b, //2n
		      T tol, int maxiter,
		      const T* A, int m, int n,
		      const T* d1, //n
		      const T* d2, //n
		      T* dxu, int& pflag, int& piter, 
		      T* r, //2n
		      T* w, //2n
		      T* p, //2n
		      T *tmpn, T *tmpm )
{
    maxiter += 2;
    if ( tol<=0. )
	tol = 1e-6;

    pcgAX<T>(dxu,A,d1,d2,w,m,n, tmpn, tmpm );

    int i;
    for ( i=0; i<n; ++i )
    {
	r[i] = b[i] - w[i];
	r[i+n] = b[i+n] - w[i+n];
	p[i] = 0.;
	p[i+n] = 0.;
    }

    bool matrix_positive_definite = true;
    T tolb_norm = Norm2<T>(b, 2*n)*tol;
    T r_norm = Norm2<T>(r, 2*n);
    T alpha = 1, tau, beta, oldtau=1;
    piter = 2;
    while ( r_norm > tolb_norm && piter < maxiter )
    {
	for ( i=0; i<n; ++i )
	{
	    tau = 1./((2+d1[i])*d1[i]-d2[i]*d2[i]);
	    w[i] = tau*(d1[i]*r[i]-d2[i]*r[n+i]); 
	    w[i+n] = tau*((2+d1[i])*r[n+i]-d2[i]*r[i]); 
	}

	tau = Xdot<T>( w, r, 2*n );
	beta = tau / oldtau;
	oldtau = tau;
	LinearVec<T>( w, p, p, 2*n, 1., beta ); //p = w+beta*p;
	
	pcgAX<T>(p,A,d1,d2,w,m,n,tmpn, tmpm );
	alpha = tau/Xdot<T>(p,w,2*n);
	if ( alpha<0 )
	    matrix_positive_definite = false;
	LinearVec<T>( dxu, p, dxu, 2*n, 1., alpha ); //dxu = dxu+alpha*p;
	LinearVec<T>( r, w, r, 2*n, 1., -alpha ); //r = r-alpha*w;
	r_norm = Norm2<T>(r, 2*n);
	piter++;
    }

    pflag = 0;
    if ( piter>maxiter-2 )
	pflag = 1;
    if ( !matrix_positive_definite )
	pflag = 3;
    piter -= 2;
}


template <typename T>
__host__ __device__ bool solveL1LS( const T* A,  //mxn
       		  const T *y, //m 
		  T *x, //n  assume initiated with 0 or something
		  T *u, //n
		  T *f, //2*n
		  T *newu, //n	//tmps
		  T *newx, //n
		  T *newf, //2*n
		  T *dxu, //2*n
		  T *z, //m
		  T *tmpm, //m
		  T *d1, //n
		  T *d2, //n
		  T *gradphi, //2*n
		  T *w, //2*n
		  T *p, //2*n
		  int m, //A.rows
		  int n, //A.cols
		  T lambda, 
		  T tar_gap, T eta, int pcgmaxi )
{
    T dobj = -mDBL_MAX;
    T s   = mDBL_MAX; 
    int pitr  = 0 ; 
    int pflg  = 0 ;

    for ( int i=0; i<n; ++i )
    {
	x[i] = 0.0;
	u[i] = 1.0;
	f[i] = -1.0; f[i+n] = -1.0;
	dxu[i] = 0.0; dxu[i+n] = 0.0;
    }

    int lsiter = 0, ntiter, i;
    T t = mMin( mMax(1,1/lambda), 2*n/1e-3 );
    T zdz, nufac, pobj, tmp, gap;
    T q1, q2, pcgtol, phi, gdx, newusum;
    bool fmaxcoefIsNeg;
    for ( ntiter=0; ntiter<MAX_NT_ITER; ++ntiter )
    {
        AxPy<T>(A, x, y, z, 1.0, -1.0, m, n, false);	
	zdz = Xdot<T>(z,z,m);
	pobj = zdz+lambda*Norm1<T>(x,n);
        
	AxPy<T>(A, z, NULL, newu, 1.0, 0, m, n, true);	
	tmp = 2.0*NormInf<T>(newu,n);
	nufac = tmp > lambda ? lambda/tmp : 1.0;

	tmp = -nufac*nufac*zdz-2*nufac*Xdot<T>(z,y,m);
	dobj  =  mMax(tmp,dobj);
	gap   =  pobj - dobj;

	if ( gap/dobj < tar_gap )
	    return true;

	if ( s >= 0.5 )
	    t = mMax( mMin(2*n*MU/gap, MU*t), t );

	AxPy<T>( A, z, NULL, newu, 2.0, 0, m, n, true );
	for ( i=0; i<n; ++i )
	{
	    q1 = 1.0/(u[i]+x[i]); 
	    q2 = 1.0/(u[i]-x[i]);
	    gradphi[i] = -newu[i]+(q1-q2)/t; 
	    gradphi[n+i] = -lambda+(q1+q2)/t; 
	    
	    q1 = q1*q1;
	    q2 = q2*q2;
	    d1[i] = (q1+q2)/t;
	    d2[i] = (q1-q2)/t;
	}

	tmp = Norm2<T>(gradphi,2*n);
	pcgtol = mMin(1e-1,eta*gap/mMin(1,tmp));

	if ( ntiter != 0 && pitr == 0 ) 
	    pcgtol = pcgtol*0.1;

	pcgSolver<T>(gradphi, pcgtol, pcgmaxi,
	       A, m, n, d1, d2, dxu, pflg, pitr, newf, w, p, newu, tmpm );

	if ( pflg==1 ) 
	    pitr = pcgmaxi; 

	phi = 0;
	for ( i=0; i<n; ++i )
	    phi += u[i];
	phi = zdz+lambda*phi;
	tmp = 0;
	for ( i=0; i<2*n; ++i )
	    tmp += log(-f[i]);
	tmp /= t;
	phi -= tmp;
	s = 1.0;
	gdx = Xdot<T>( gradphi, dxu, 2*n ) * ALPHA * (-1);
	for ( lsiter=1; lsiter<=MAX_LS_ITER; ++lsiter )
	{
	    fmaxcoefIsNeg = true;
	    newusum = 0.0;
	    for ( i=0; i<n; ++i )
	    {
		newx[i] = x[i]+s*dxu[i]; 
    		newu[i] = u[i]+s*dxu[n+i];
		newusum += newu[i];

		newf[i] = newx[i] - newu[i];
		newf[n+i] = -newx[i] - newu[i];
		if ( newf[i] >=0 || newf[n+i] >=0 )
		    fmaxcoefIsNeg = false;
	    }
	    if ( fmaxcoefIsNeg )
	    {
		AxPy<T>(A, newx, y, z, 1, -1, m, n, false); //newz	
		tmp = 0.0;
		for ( i=0; i<n; ++i )
		{
		    tmp += log(-newf[i]);
		    tmp += log(-newf[i+n]);
		}
		tmp /= t;
		tmp = Xdot<T>(z,z,m) + lambda*newusum - tmp - phi;
		if ( tmp <= s*gdx )
		    break;
	    }

	    s = BETA*s;
	}

	if ( lsiter==MAX_LS_ITER ) 
	    break; 

	for ( i=0; i<n; ++i )
	{
    	    x[i] = newx[i]; 
	    u[i] = newu[i]; 
	    f[i] = newf[i];
	    f[i+n] = newf[i+n];
	}
    }

    return true;
}

#endif //_CUDA_L1LS_H_
