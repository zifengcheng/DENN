alpha=$1
temp=$2
k=$3
for ((seed=2; seed<3; ++seed)); do
    python run_baselines.py --random_seed $seed --dataset_name AAPD --root ../DataSet/AAPD --setting F --alpha ${alpha} --temp ${temp} --k ${k} --gpu 0,1 --val_iter 200
    python test.py --random_seed $seed --dataset_name AAPD --root ../DataSet/AAPD --setting F --alpha ${alpha} --temp ${temp} --k ${k} --gpu 0,1,2,3 --batch_size 128
done
