#!/bin/bash
run_sim(){
  echo "Runing simulation with horizon $1 ..."
  python main.py --horizon $1 --method 'gradient' > output_horizon_$1.log
}

rm -rf output*.log
export OMP_NUM_THREADS=2

for i in 5 10 15 20 25
do
	run_sim $i & # Put a function in the background
done
wait 
echo "All done"