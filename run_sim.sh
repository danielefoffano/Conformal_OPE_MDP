#!/bin/bash
run_sim(){
  echo "Runing simulation with horizon $1 ..."
  python main.py --horizon $1 --method 'gradient' --runs 6 7 8 9 10 11 > output_horizon_$1.log
}

rm -rf output*.log
export OMP_NUM_THREADS=2

for i in 5
do
	run_sim $i & # Put a function in the background
done
wait 
echo "All done"