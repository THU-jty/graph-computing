# !/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable>" >&2
    exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW3/data

EXECUTABLE=$1
REP=64

FILELIST=`ls -Sr ${DATAPATH} | grep "\.csr"`

for FILE in ${FILELIST}; do
    if test -f ${DATAPATH}/${FILE}; then
        srun -p gpu ${EXECUTABLE} bfs  ${REP} ${DATAPATH}/${FILE}
        srun -p gpu ${EXECUTABLE} sssp  ${REP} ${DATAPATH}/${FILE}
    fi
done
