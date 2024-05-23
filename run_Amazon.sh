alpha=$1
temp=$2
k=$3
threshold=$4
for ((seed=0; seed<1; ++seed)); do
    python run_baselines.py --random_seed $seed --dataset_name Amazon --root ../DataSet/Amazon --setting F --alpha ${alpha} --temp ${temp} --k ${k} --gpu 0,1 --val_iter 500
    python test_item_model.py --random_seed $seed --dataset_name Amazon --root ../DataSet/Amazon --setting F --alpha ${alpha} --temp ${temp} --k ${k} --gpu 0,1,2,3 --batch_size 128 --thresh ${threshold}
done
