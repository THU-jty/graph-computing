# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/course/HW3/data

EXECUTABLE=$1
REP=64

srun -p gpu ${EXECUTABLE} bfs  ${REP} ${DATAPATH}/human_gene1.csr
srun -p gpu ${EXECUTABLE} sssp ${REP} ${DATAPATH}/human_gene1.csr
