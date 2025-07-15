#ifndef _CUDA_PROCESSKERNEL_H_
#define _CUDA_PROCESSKERNEL_H_

#include "utilities.h"

bool process(const IOPar::SetupInfo& su,const std::vector<int>& h0,
	     const std::vector<int>& h1,std::string& err);

void testL1LS();

#endif /*_CUDA_PROCESSKERNEL_H_*/
