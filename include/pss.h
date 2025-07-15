#pragma once

#include "../lib/Eigen/Core"


using namespace Eigen;

struct BGModel 
{
    VectorXd	vp;
    VectorXd	vs;
    VectorXd	den;
};


struct WellData 
{
    VectorXd	vp;
    VectorXd	vs;
    VectorXd	den;
    VectorXd	vpr;
    VectorXd	vsr;
    VectorXd	denr;
    int		cdpid;
    int		maxiter;
    int		iterdone;
    double	err;
    double	corr; //corelation between seis and synthetic
};


struct InvPar
{
    MatrixXd	waveformmatrix_compact;
    VectorXd	angdata;
    BGModel	bmodel;
    VectorXi	id_index;
    int		nsmooth;
    double	pi_c;
};
    

class PreStackSim
{
public:    
    			PreStackSim()	{}

    void		simulate(const MatrixXd& s_data,const MatrixXd& w_data,
				 const VectorXd& ang,const BGModel& bmodel,
				 WellData& res,
				 VectorXi id_flag=VectorXi::Zero(1),
				 VectorXi target_flag = VectorXi::Zero(1),
				 int nsmooth = 20,
				 double pi_c = 1.4,
				 bool display = true);
    static double 	prestackOptFun(const VectorXd& x,const InvPar& d);
    static MatrixXd	toeplitz(const VectorXd& wltmatrix,int ncol);

protected:
    MatrixXd		extractCols(const MatrixXd& D,
	    			    const VectorXi& index);
    MatrixXd		extractRows(const MatrixXd& D,
	    			    const VectorXi& index);
    VectorXd		extractItems(const VectorXd& D,
	    			     const VectorXi& index);
    static VectorXd	velFromReflectivity(const VectorXd&r,double v0);
    static void		prestackForward(const VectorXd& x,
	    				const InvPar& d,WellData& wd);
    static VectorXd	movemean(const VectorXd& a,int len);
    void		syntheticModeling(const MatrixXd& s_data,
	    				  const MatrixXd& w_data,
	 				  const VectorXd& ang,WellData& bmodel);
};





