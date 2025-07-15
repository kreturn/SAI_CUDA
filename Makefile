# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda-6.5
CUDA_INSTALL_DIR=/usr/local/cuda-6.5

# CC compiler options:
CC=g++
CC_FLAGS=-Wall -DDEBUG -g #-O3 -fopenmp -lgomp
#CC_FLAGS=-Wall -O3 -fopenmp -lgomp
CC_LIBS=

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -arch=sm_35 -g -G
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcublas -lcufft 

## Project file structure ##

# Source file directory:
SRC_DIR=src

# Object file directory:
OBJ_DIR=lib

# Include header file diretory:
INC_DIR=include

# Target executable name:
EXE = saicu

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/processkernel.o $(OBJ_DIR)/utilities.o 


## Compile ##
# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ -Ilib/Eigen $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)



