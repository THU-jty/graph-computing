# !/bin/bash

DATAPATH=/home/course/HW3/data
STATFILE=stat.csv

FILELIST=`ls -Sr ${DATAPATH} | grep "\.csr"`

FIRST=1

for FILE in ${FILELIST}; do
    if test -f ${DATAPATH}/${FILE}; then
        if [ $FIRST -eq 1 ]; then
            FIRST=0
            ./genstat ${DATAPATH}/${FILE} ${STATFILE} -h
        else
            ./genstat ${DATAPATH}/${FILE} ${STATFILE}
        fi
    fi
done

