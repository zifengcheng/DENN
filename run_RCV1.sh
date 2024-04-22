alpha=$1
temp=$2
beta=$3
k=$4
for ((seed=2; seed<3; ++seed)); do
    python run_baselines.py --random_seed $seed --dataset_name RCV1-V2 --root ../DataSet/RCV1-V2 --setting F --alpha ${alpha} --temp ${temp} --beta ${beta} --k ${k} --val_iter 2000
    python test.py --random_seed $seed --dataset_name RCV1-V2 --root ../DataSet/RCV1-V2 --setting F --alpha ${alpha} --temp ${temp} --beta ${beta} --k ${k} --gpu 0,1,2,3 --batch_size 128
done
