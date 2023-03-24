#!/bin/bash
# run_sim(){
#   echo "Runing simulation with horizon $1 ..."
#   python main.py --horizon $1 --method 'empirical' > output_horizon_$1.log
# }

# rm -rf output*.log
# #export OMP_NUM_THREADS=4

# for i in 5 10 15 20 25
# do
# 	run_sim $i & # Put a function in the background
# done

# python main.py --horizon 25 --method 'empirical' --seed 49213 -r 20 21 22 23 24 > output_horizon_25_3.log &
# python main.py --horizon 20 --method 'empirical' --seed 597213 -r 20 21 22 23 24 > output_horizon_20_3.log &
# python main.py --horizon 15 --method 'empirical' --seed 974213 -r 20 21 22 23 24 > output_horizon_15_3.log &

# python main.py --horizon 40 --method 'empirical' --seed 90860013 -r 25 26 27 28 29 > output_horizon_40_0.log &
# python main.py --horizon 40 --method 'empirical' --seed 5849444 -r 20 21 22 23 24 > output_horizon_40_1.log &
python main.py --horizon 15 --method 'empirical' --seed 1111111 -r 17 18 19 > output_horizon_15_last.log &
python main.py --horizon 25 --method 'empirical' --seed 1111112 -r 12 13 14 > output_horizon_25_last.log &
python main.py --horizon 40 --method 'empirical' --seed 1111113 -r 15 16 17 18 19  > output_horizon_40_10.log &
python main.py --horizon 40 --method 'empirical' --seed 1111114 -r 10 11 12 13 14 > output_horizon_40_11.log &
python main.py --horizon 40 --method 'empirical' --seed 1111115 -r 5 6 7 8 9 > output_horizon_40_12.log &
python main.py --horizon 40 --method 'empirical' --seed 1111116 -r 0 1 2 3 4 > output_horizon_40_13.log &
python main.py --horizon 40 --method 'empirical' --seed 1111117 -r 21 26 27 28 29 > output_horizon_40_14.log &

wait 
echo "All done"
