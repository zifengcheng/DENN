alpha=$1
temp=$2
beta=$3
k=$4
threshold=$5
for ((seed=0; seed<1; ++seed)); do
    python run_baselines.py --random_seed $seed --dataset_name Amazon --root ../DataSet/Amazon --setting F --alpha ${alpha} --temp ${temp} --beta ${beta} --k ${k} --gpu 0,1 --val_iter 500
    python test_item_model.py --random_seed $seed --dataset_name Amazon --root ../DataSet/Amazon --setting F --alpha ${alpha} --temp ${temp} --beta ${beta} --k ${k} --gpu 0,1,2,3 --batch_size 128 --thresh ${threshold}
done
