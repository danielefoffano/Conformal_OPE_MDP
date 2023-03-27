#!/bin/bash
for j in {20,25,40}
do
	for i in {20..29}
	do
		sbatch -o Conformal_OPE_gradient_H_$j-$i.out runjob.sh $j gradient $i
	done
done