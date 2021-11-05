
CXX=g++

CXX_FLAGS=-std=c++17 -O3 -g -Wall -march=native -DNDEBUG

SRILM="/home/gerstenberger/rwth/work/repo/srilm"

OUT_DIR=bin
SRC_DIR=src

INCLUDES=$(shell python3-config --includes) -Iextern/pybind11/include -I$(SRILM)/include

LIB_DIRS=-L$(SRILM)/lib/i686-m64
LIBS=-loolm -ldstruct -lmisc -lz

PY_EXT=$(shell python3-config --extension-suffix)

srilm_binder:
	$(CXX) $(CXX_FLAGS) \
	$(INCLUDES) -I$(SRC_DIR)/ \
	-shared -fPIC -fopenmp \
	$(SRC_DIR)/binder.cxx \
	-o ${OUT_DIR}/python/srilm$(PY_EXT) \
	$(LIB_DIRS) $(LIBS)

prepare:
	mkdir -p $(OUT_DIR)/python
	mkdir -p $(OUT_DIR)/test

clean:
	rm -rf $(OUT_DIR)
