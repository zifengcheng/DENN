alpha=$1
temp=$2
beta=$3
k=$4
for ((seed=0; seed<1; ++seed)); do
    python run_baselines.py --random_seed $seed --dataset_name EUR --root ../DataSet/EUR --setting F --alpha ${alpha} --temp ${temp} --beta ${beta} --k ${k} --gpu 0,1 --val_iter 2000 --max_epochs 100
    python test.py --random_seed $seed --dataset_name EUR --root ../DataSet/EUR --setting F --alpha ${alpha} --temp ${temp} --beta ${beta} --k ${k} --gpu 0,1,2,3 --batch_size 128  --threshold 0.9
done
