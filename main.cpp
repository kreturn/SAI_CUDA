#include <iostream>
#include <sys/stat.h>
#include "include/utilities.h"
#include "include/processkernel.cuh"

using namespace std;

#define mErrReturn(msg) \
    cout << "Error: " << msg << endl; return -1



int main( int argc, char** argv )
{
    testL1LS(); return 0; //Testing
    if ( argc<2 )
    {
	cout << "Run sai and oput in whole files (instead each cdp)" << endl; 
	mErrReturn("Run executable as: ./saiw config.par");
    }

    std::string parfile(argv[1]);

    IOPar par;
    par.getPar( parfile.c_str() );
    const IOPar::SetupInfo& su = *par.setupInfo();

    std::vector<int> h0, h1; 
    if ( !su.horinp.empty() )
    {
    	if ( !Utilities::readHorizonConstrainsInMs(su.horinp.c_str(),h0,h1) )
	{
	    cout << "Error: could not read horizon file " 
		 << su.horinp.c_str() << endl; 
	    return -1;
	}
    
	if ( h0.size() != su.nrcdp )
    	{
    	    mErrReturn("Horizon size doesn't match with configure cdp size");
    	}

	const int minh = *std::min_element(h0.begin(), h0.end());
	if ( minh < su.t0 )
	{
    	    mErrReturn("Minimum horizon time is not in data time range");
	}
	const int maxh = *std::max_element(h1.begin(), h1.end());
	if ( maxh > su.t1 )
	{
    	    mErrReturn("Maximum horizon time is not in data time range");
	}
    }

    if ( su.jobdir.empty() )
    {
	mErrReturn("Please define a processing jobdir in configure file");
    }

    if ( su.aginp.empty() )
    {
	mErrReturn("Please define angle gather file in configure file");
    }

    if ( su.wvinp.empty() )
    {
	mErrReturn("Please define a wavelet file in configure file");
    }

    if ( su.vpinp.empty() )
    {
	mErrReturn("Please define input Vp in configure file");
    }

    if ( su.vsinp.empty() )
    {
	mErrReturn("Please define input Vs in configure file");
    }
    
    if ( su.deninp.empty() )
    {
	mErrReturn("Please define input Density in configure file");
    }
    
    const size_t startidx = su.startcdp <= 0 ? 0 : su.startcdp-1;
    const size_t stopidx = su.stopcdp <= 0 || su.stopcdp > su.nrcdp ? 
	su.nrcdp-1 : su.stopcdp-1;

    time_t tstart = time(0);
    cout<<"-----Simulated Annealing Inversion Started--------"<<endl;
    cout<<"-----Start time: "<<ctime(&tstart)<<endl;
    cout<<"-----Total number of CDPs: "<<su.nrcdp<<endl;
    cout<<"-----Node CDP range ("<<startidx+1<<"--"<<stopidx+1<< ")"<<endl;

    struct stat info;
    if ( stat(su.jobdir.c_str(), &info)!=0 )
    {
	std::string cmd( "mkdir " ); cmd.append( su.jobdir.c_str() );
	system( cmd.c_str() );
    }
    else if ( su.cljobdir )
    {
	std::string cmd( "rm -rf 0*" ); 
	cmd.append( su.jobdir.c_str() );
	cmd.append( "/*" );
	system( cmd.c_str() );
    }

    std::string err;
    if ( !process( su, h0, h1, err ) )
    {
	cout << "Error: " << err.c_str() << endl;
	return -1;
    }

    time_t tend = time(0);
    float dlibtime = difftime(tend,tstart)/60;
    cout<<"----------SAI Processing done--------"<<endl;
    cout<<"----------Current time: "<<ctime(&tend)<<endl;
    cout<<"----------Total running minutes is: "<<dlibtime<<endl;

    return 0;
}
