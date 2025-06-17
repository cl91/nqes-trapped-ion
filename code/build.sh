#!/bin/bash

EIGEN_INC=${EIGEN_INC:=/usr/include/eigen3}
PYTHON=${PYTHON:=python}

OPT=0
buildty="debug"

if [[ $1 == "opt" ]]; then
    OPT=1
    buildty="opt"
fi

OUTDIR=$PWD
cd "$(dirname "$0")"
GITHASH=$(git show -s --format="%h (%ci)" HEAD)
COMMONOPTS="-std=c++23 -Wall -Wno-uninitialized -I${EIGEN_INC} -g -rdynamic -fexceptions -funwind-tables -fasynchronous-unwind-tables -Wno-maybe-uninitialized"
LINKLIBS="-lmatio"
DBGOPTS="-DDBG=1 -DDEBUG -g -fsanitize=address"
NDBGOPTS="-DNDEBUG -O3 -march=native -mtune=native -Wno-unused-variable"
if (( $OPT )); then
    COMMONOPTS+=" $NDBGOPTS"
else
    COMMONOPTS+=" $DBGOPTS"
fi
COMPILE_CMD="c++ -DNOGPU $COMMONOPTS -march=native -fopenmp"

PARAMS=("8 16 2" "8 32 2" "16 32 2" "16 64 2" "32 64 2" "32 128 2" "64 128 2" "64 256 2" "128 256 2")

for params in "${PARAMS[@]}"; do
    N=$(echo $params | cut -f1 -d' ')
    M=$(echo $params | cut -f2 -d' ')
    K=$(echo $params | cut -f3 -d' ')
    PARAMDEF="-DNUM_SPINS=$N -DNUM_HIDDEN_UNITS=$M -DNUM_EIGENSTATES=$K"
    $PYTHON load_jmat.py $N > "Jmat${N}.h"
    if (( $N <= 16 )); then
	$COMPILE_CMD $PARAMDEF -DGITHASH="\"$GITHASH\"" -DJMAT_INC="\"Jmat${N}.h\"" \
		     rbm.cc $LINKLIBS -o $OUTDIR/rbm-N$N-M$M-K$K-$buildty
    else
	$COMPILE_CMD $PARAMDEF -DMATFREE=1 -DGITHASH="\"$GITHASH\"" -DJMAT_INC="\"Jmat${N}.h\"" \
		     rbm.cc $LINKLIBS -o $OUTDIR/rbm-matfree-N$N-M$M-K$K-$buildty
    fi
done
