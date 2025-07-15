#include <iostream>
#include <string>
#include <ctime>
#include <algorithm>
#include <iterator>
#include <vector>
#include <cmath>
#include <numeric>      // std::accumulate
#include "l1ls.h"
#include "pss.h"
#include "utilities.h"
#include "sa.h"

#define PI 3.1415926535897

using namespace std;
using namespace Eigen;


class CompareFun
{
    public: 
	CompareFun(const std::vector<double>& v) : vec(v) {} 
	bool operator() (int i, int j) { return vec[i] < vec[j]; }
	const std::vector<double>& vec;
};


class MyOpt: public Optim
{
public:

    MyOpt( const InvPar& ip )
        : ip_(ip),dx_(1e-7) {}

    void setDx(double v)	{ if ( v>0 ) dx_=v; }
    double functionValue(const VectorXd& x)
    { 
	return PreStackSim::prestackOptFun(x,ip_);
    }

    void functionGrad(const VectorXd& x,VectorXd& grad)
    {
	const double inv2dx = 1./(2*dx_);
	grad.resize( x.size() );
	for ( int i=0; i<x.size(); ++i )
	{
	    VectorXd vi = x;
	    vi(i) = vi(i)-dx_;
	    const double fm = functionValue(vi);
	    vi(i) = vi(i)+2*dx_;
	    const double fp = functionValue(vi);
	    grad(i) = (fp-fm)*inv2dx;
	}
    }

private:

    const InvPar& ip_;
    double	dx_;
};


MatrixXd PreStackSim::toeplitz( const VectorXd& wltmatrix, int ncol )
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


MatrixXd PreStackSim::extractCols( const MatrixXd& mat, 
				   const VectorXi& index )
{
    const int ncol = index.size();
    const int nrow = mat.rows();
    MatrixXd matsub(nrow, ncol);

    for ( int i = 0; i < ncol; i++ )
    {
	if ( index[i] < mat.cols()-1 )
	    matsub.col(i) = mat.col(index[i]);
    }
    
    return matsub;
}


MatrixXd PreStackSim::extractRows( const MatrixXd& D, 
				   const VectorXi& index )
{
    const int nrow = index.size();
    const int ncol = D.cols();
    
    MatrixXd Dsub(nrow, ncol);
    for ( int i=0; i<nrow; i++ )
    {
	if ( index[i]<D.rows()-1 )
	    Dsub.row(i) = D.row(index[i]);
    }

    return Dsub;
}


VectorXd PreStackSim::extractItems( const VectorXd& D, 
				    const VectorXi& index )
{
    const int n = index.size();
    VectorXd Dsub(n);

    for ( int i=0; i<n; i++ )
    {
	if ( index[i]<D.rows()-1 )
	    Dsub(i) = D(index[i]);
    }

    return Dsub;
}


VectorXd PreStackSim::movemean( const VectorXd& a, int len )
{
    const int nsamp = a.size();
    const int half_len = floor(len/2);
    VectorXd amean( nsamp );

    for ( int i=0; i < nsamp; i++ )
    {
	//int n_midpt = ceil((i + 1) / 2);
	int n_midpt = i+1;
	int istart, iend;
	if ( len%2 == 0 )
	{
	    istart = std::max( (n_midpt - 1) - (half_len - 1), 1 );
	    iend = std::min( n_midpt + (half_len - 1), nsamp );
	}
	else
	{
	    istart = std::max(n_midpt - half_len, 1);
	    iend = std::min(n_midpt + half_len, nsamp);
	}

	double tmp = std::accumulate( a.begin()+istart-1, a.begin()+iend, 0.0);
	amean[i] = tmp/(iend - istart + 1);
    }

    return amean;
}


VectorXd PreStackSim::velFromReflectivity( const VectorXd& r, double vp0 )
{
    const int nsamp = r.size();
    VectorXd vp(nsamp);
    vp(0) = vp0;

    for ( int i=0; i<nsamp-1; i++ )
	vp(i+1) = (2+r(i)) / (2-r(i)) * vp(i);

    return vp;
}


double PreStackSim::prestackOptFun( const VectorXd& x, const InvPar& da )
{
    const int nsamp = da.bmodel.vp.size();
    const int nlayer = da.id_index.size();

    VectorXd vp_r0 = x.segment(0, nlayer);
    VectorXd vs_r0 = x.segment(0,nlayer).array() * 
		     x.segment(nlayer, nlayer).array();
    VectorXd den_r0 = x.segment(0,nlayer).array() * 
		      x.segment(2*nlayer, nlayer).array();

    VectorXd vp1_r = VectorXd::Zero(nsamp);
    VectorXd vs1_r = VectorXd::Zero(nsamp);
    VectorXd den1_r = VectorXd::Zero(nsamp);
    for ( int i=0; i<da.id_index.size(); ++i )
    {
	vp1_r(da.id_index(i)) = vp_r0(i);
	vs1_r(da.id_index(i)) = vs_r0(i);
	den1_r(da.id_index(i)) = den_r0(i);
    }
    
    VectorXd sol_compact(3 * nlayer, 1);
    sol_compact.segment(0, nlayer) = vp_r0;
    sol_compact.segment(nlayer, nlayer) = vs_r0;
    sol_compact.segment(2 * nlayer, nlayer) = den_r0;

    VectorXd data_modeled = da.waveformmatrix_compact * sol_compact;
    
    VectorXd vp1 = velFromReflectivity(vp1_r, da.bmodel.vp[0]);
    VectorXd vs1 = velFromReflectivity(vs1_r, da.bmodel.vs[0]);
    VectorXd den1 = velFromReflectivity(den1_r, da.bmodel.den[0]);

    VectorXd datadiff = data_modeled - da.angdata;
    const double err1 = datadiff.norm() / da.angdata.norm();

    const int nsmooth = da.nsmooth;
    VectorXd vp_bac1 = movemean(vp1, nsmooth);
    VectorXd vp_bac2 = movemean(vp_bac1, nsmooth);
    
    VectorXd vs_bac1 = movemean(vs1, nsmooth);
    VectorXd vs_bac2 = movemean(vs_bac1, nsmooth);

    VectorXd den_bac1 = movemean(den1, nsmooth);
    VectorXd den_bac2 = movemean(den_bac1, nsmooth);
    
    VectorXd vp_err(nsamp), vs_err(nsamp), den_err(nsamp), pi_err(nsamp);
    for ( int i=0; i<nsamp; ++i )
    {
	vp_err(i) = (vp_bac2[i] - da.bmodel.vp[i]) / da.bmodel.vp[i];
	vs_err(i) = (vs_bac2[i] - da.bmodel.vs[i]) / da.bmodel.vs[i];
	den_err(i) = (den_bac2[i] - da.bmodel.den[i]) / da.bmodel.den[i];
	double pi_bac2 = vp_bac2[i] * den_bac2[i] - 
	    da.pi_c * vs_bac2[i] * den_bac2[i];
	double pi_bac = da.bmodel.vp[i] * da.bmodel.den[i] - 
	    da.pi_c * da.bmodel.vs[i] * da.bmodel.den[i];
	pi_err(i) = (pi_bac2 - pi_bac) / pi_bac;
    }

    double err2 = Utilities::norm2(vp_err) + Utilities::norm2(vs_err) + 
		 Utilities::norm2(den_err) + 4*Utilities::norm2(pi_err);
    return err1+0.5*err2;
}


void PreStackSim::prestackForward( const VectorXd& x, 
				   const InvPar& invpar, WellData& wd )
{
    int nsamp = invpar.bmodel.vp.size();
    int nlayer = invpar.id_index.size();

    VectorXd vp_r0 = x.segment(0, nlayer);
    VectorXd vs_r0 = x.segment(0,nlayer).array() * 
		     x.segment(nlayer, nlayer).array();
    VectorXd den_r0 = x.segment(0,nlayer).array() * 
		      x.segment(2*nlayer, nlayer).array();

    VectorXd vp1_r = VectorXd::Zero(nsamp);
    VectorXd vs1_r = VectorXd::Zero(nsamp);
    VectorXd den1_r = VectorXd::Zero(nsamp);
    for ( int i=0; i<invpar.id_index.size(); ++i )
    {
	vp1_r(invpar.id_index(i)) = vp_r0(i);
	vs1_r(invpar.id_index(i)) = vs_r0(i);
	den1_r(invpar.id_index(i)) = den_r0(i);
    }

    VectorXd sol_compact(3*nlayer, 1);
    sol_compact.segment(0, nlayer) = vp_r0;
    sol_compact.segment(nlayer, nlayer) = vs_r0;
    sol_compact.segment(2*nlayer, nlayer) = den_r0;

    VectorXd data_modeled = invpar.waveformmatrix_compact * sol_compact;

    wd.vp = velFromReflectivity(vp1_r, invpar.bmodel.vp[0]);
    wd.vs = velFromReflectivity(vs1_r, invpar.bmodel.vs[0]);
    wd.den = velFromReflectivity(den1_r, invpar.bmodel.den[0]);
}


void PreStackSim::simulate( const MatrixXd& s_data, const MatrixXd& w_data, 
			    const VectorXd& ang, const BGModel& bmodel, 
			    WellData& res, VectorXi id_flag,
			    VectorXi target_flag, int nsmooth, 
			    double pi_c, bool display )
{
    display = false;
    const int nsamp = bmodel.vp.size();
    const int nrangles = w_data.cols();
    int nspike = floor(nsamp/4);

    if ( target_flag.rows()<nsamp )
    {
	target_flag.resize(nsamp);
	target_flag.setZero();
	nspike = floor(nsamp/3);
    }

    VectorXd vsvpra(nsamp);
    for ( int i=0; i<nsamp; i++ )
	vsvpra(i) = bmodel.vs[i] / bmodel.vp[i];

    MatrixXd refmatrix = MatrixXd::Constant( nrangles*nsamp, nsamp*3, 0.0 );
    MatrixXd waveformmatrix = MatrixXd::Zero( nrangles*nsamp, nsamp*3 );
    VectorXd angdata = VectorXd::Zero( nrangles*nsamp );

    const double deg2rad = M_PI/180.0;

    for (int i=0; i<nrangles; ++i )
    {
	const double angr = ang(i) * deg2rad;
	const VectorXd& wlt = w_data.col(i);
	double  normval = wlt.norm();
	VectorXd tempwlt = wlt/normval;
	const int nshift = floor( wlt.rows()*0.5 );

	for ( int j=0; j<nsamp; ++j )
	{
	    refmatrix(nsamp*i+j,j) = 0.5 * ( 1 + pow(tan(angr), 2) );
	    refmatrix(nsamp*i+j,j+nsamp) = -4 * pow(vsvpra(j)*sin(angr),2);
	    refmatrix(nsamp*i+j,j+2*nsamp) = 0.5-2*pow(vsvpra(j)*sin(angr),2);  
	}

	MatrixXd W_extend = toeplitz( tempwlt, nsamp );
	MatrixXd W = W_extend.block( nshift, 0, nsamp, nsamp );
	MatrixXd ref0 = refmatrix.block(nsamp*i, 0, nsamp, nsamp*3);
	waveformmatrix.block(nsamp*i, 0, nsamp, nsamp*3) = W * ref0;
	angdata.segment(nsamp*i, nsamp) = s_data.col(i) / normval;
    }

    if ( id_flag.rows()<nsamp )
    {
	clock_t begin1 = clock();
	if ( display )
	    cout << "Starting L1 minimization to select the layers ..." << endl;

	id_flag.resize(nsamp);
	id_flag.setZero();
	
	VectorXd temp = 2*waveformmatrix.transpose()*angdata;
	const double lambda = 0.0001*temp.cwiseAbs().maxCoeff();

	VectorXd x;
	L1LS::History hist;
	L1LS::solve( waveformmatrix, angdata, x, hist, lambda, 0, 1.0e-4 );

	VectorXd vp_r1 = x.segment(0, nsamp);
	VectorXd vs_r1 = x.segment(nsamp, nsamp);
	VectorXd den_r1 = x.segment(2*nsamp, nsamp);
    
	res.vpr = vp_r1;
	res.vsr = vs_r1;
	res.denr = den_r1; 

	VectorXd absref0 = vp_r1.cwiseAbs()+vs_r1.cwiseAbs()+den_r1.cwiseAbs();
/*std::vector<double> absr(absref0.data(),absref0.data()+absref0.size());
	std::vector<int> id_sort( absr.size() );
	for ( size_t i=0; i<absr.size(); ++i )
	    id_sort[i] = i;
	sort(id_sort.begin(), id_sort.end(),CompareFun(absr));

	std::reverse( id_sort.begin(), id_sort.end() ); TODO */
	VectorXi id_sort = VectorXi::Zero( absref0.size() );
	int ispike = 0;
	int id_check = 1;
	VectorXi sel_id = VectorXi::Constant(nspike+1,-999);

	while ( ispike <= nspike && id_check < nsamp )
	{
	    VectorXi id_dist = sel_id - 
		VectorXi::Constant( sel_id.size(), id_sort[id_check - 1]) ;
	    if ( id_dist.cwiseAbs().minCoeff() > 1 || ispike == 0 )
	    {
		ispike++;
		sel_id(ispike-1) = id_sort[id_check - 1];
	    }
	    ++id_check;
	}
	
	for (int i = 0; i < sel_id.size(); i++)
	    id_flag(sel_id(i)) = 1;

	if ( display )
	{
	    clock_t end1 = clock();
	    double elapsed_secs = double(end1-begin1) / CLOCKS_PER_SEC;
	    cout << "Total time used for l1 minimization: " << elapsed_secs << 
		" seconds" << endl;
	}
    }

    VectorXi id_index;
    for ( int i = 0; i<nsamp-1; i++ )
    {
	if ( target_flag(i)>0 || id_flag(i) > 0 )
	    id_index << i;
    }
    
    VectorXd vp_r_bac(nsamp);
    for ( int i=0; i<nsamp-1; ++i )
	vp_r_bac(i) = (bmodel.vp[i+1]-bmodel.vp[i]) / 
	    (bmodel.vp[i+1] + bmodel.vp[i]) * 2;
    vp_r_bac(nsamp-1) = 0;
    VectorXd vp_r0 = extractItems(vp_r_bac, id_index);
    int nlayer = vp_r0.rows();
    
    //clock_t begin2 = clock();
    if ( display )
    { 
	cout << "Starting L2 optimization process, layers are selected: " 
	     << nlayer << endl;
    }	

    VectorXi id_index3 = id_index;
    for ( int i=0; i<nlayer; i++ )
	id_index3 << (id_index[i]+nsamp);

    for ( int i=0; i<nlayer; i++ )
	id_index3 << (id_index[i]+2*nsamp);

    MatrixXd waveformmatrix_compact( nrangles*nsamp, nlayer*3 );
    waveformmatrix_compact = extractCols( waveformmatrix, id_index3 );

    VectorXi id_index1( id_index.size() );
    for ( size_t i=0; i<id_index.size(); ++i )
	id_index1(i) = id_index[i];

    InvPar invpar;
    invpar.waveformmatrix_compact = waveformmatrix_compact;
    invpar.angdata = angdata;
    invpar.bmodel = bmodel;
    invpar.id_index = id_index1;
    invpar.nsmooth = nsmooth;
    invpar.pi_c = pi_c;
    
    VectorXd vones = VectorXd::Ones(nlayer);
    VectorXd x(3*nlayer);  x << vp_r0, vones, vones;
    VectorXd lb(3*nlayer); lb << -0.3*vones, 0.5*vones, 0.5*vones;
    VectorXd ub(3*nlayer); ub << 0.3*vones, 1.5*vones, 1.5*vones;

    MyOpt mo(invpar);
    AlgoSearchOpt ase;
    AlgoStopOpt ast(1e-8,res.maxiter,0.5);
    //ast.turnOnVerbose();
    res.err = mo.executeConstrained( ase, ast, x, res.iterdone, lb, ub );

    /*    
    double err = prestackOptFun(x,invpar);
    res.err = err;    
    if ( display )
    {
	clock_t end2 = clock();
	double elapsed_secs = double(end2 - begin2) / CLOCKS_PER_SEC;
	cout << "Optimization is complete, it takes " << elapsed_secs 
	    << " seconds" << endl;
	//cout << "The final err is: " << err << endl;
    }*/
    
    prestackForward( x, invpar, res );
    syntheticModeling(s_data,w_data,ang,res);
}

void PreStackSim::syntheticModeling( const MatrixXd& s_data,
	const MatrixXd& w_data, const VectorXd& ang, WellData& invdata )
{
    const int nrangles = ang.size();
    const int nsamp = invdata.vp.size();
    
    VectorXd vp = Map<VectorXd>( invdata.vp.data(), nsamp );
    VectorXd vs = Map<VectorXd>( invdata.vs.data(), nsamp );
    VectorXd den = Map<VectorXd>( invdata.den.data(), nsamp );

    MatrixXd s_matrix = MatrixXd::Zero( nsamp, nrangles );
    if ( nrangles==1 )
    {
	VectorXd ai = vp.array()*den.array();
	VectorXd r_matrix = VectorXd::Zero(nsamp);
	for ( int i=0; i<nsamp-1; ++i )
	    r_matrix(i) = (ai[i+1]-ai[i])/(ai[i+1]+ai[i])*2;
	    
	VectorXd s( nsamp );
	Utilities::convz<double>( r_matrix, w_data.col(0), s );
	s_matrix.col(0) = s;
    }
    else
    {
	VectorXd vsvpra = vs.array()/vp.array();
	VectorXd vsvpra2 = vsvpra.array()*vsvpra.array();
        VectorXd vp_r = VectorXd::Zero(nsamp);
        VectorXd vs_r = VectorXd::Zero(nsamp);
	VectorXd den_r = VectorXd::Zero(nsamp);
	for ( int i=0; i<nsamp-1; ++i )
	{
	    vp_r[i] = (vp[i+1]-vp[i])/(vp[i]+vp[i+1])*2;
	    vs_r[i] = (vs[i+1]-vs[i])/(vs[i]+vs[i+1])*2;
	    den_r[i] = (den[i+1]-den[i])/(den[i]+den[i+1])*2;
	}

	for ( int i=0; i<nrangles; i++ )
	{
	    double angr = ang[i]*M_PI/180;
	    double a = 0.5*(1+pow(tan(angr),2));
	    VectorXd b = (-4*pow(sin(angr),2))*vsvpra2.array();
	    VectorXd r = a * vp_r.array() + b.array() * vs_r.array() +
		0.5 * den_r.array() - 
		(2*pow(sin(angr),2))*vsvpra2.array()*den_r.array();

	    VectorXd s;	    
	    Utilities::convz<double>( r, w_data.col(i), s );
            s_matrix.col(i) = s;
	}
    }

    VectorXd corr_list = VectorXd::Zero(nrangles);
    for ( int i=0; i<nrangles; ++i )
    {
        corr_list[i] = Utilities::pearsonCorrelation<double>( 
		s_matrix.col(i), s_data.col(i) );
    }

    invdata.corr = corr_list.mean();
}
