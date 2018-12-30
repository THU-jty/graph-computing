# !/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <executable> <number of nodes> <process per node>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW3/data

EXECUTABLE=$1
NODE_NUM=$2
PROC_NUM=$(($2*$3))
REP=64

srun -N ${NODE_NUM} -n ${PROC_NUM} ${EXECUTABLE} bfs  ${REP} ${DATAPATH}/human_gene1.csr
srun -N ${NODE_NUM} -n ${PROC_NUM} ${EXECUTABLE} sssp ${REP} ${DATAPATH}/human_gene1.csr
