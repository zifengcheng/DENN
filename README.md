# A Debiased Nearest Neighbors Framework for Multi-Label Text Classification
Code of Our Paper "A Debiased Nearest Neighbors Framework for Multi-Label Text Classification"


## Enviroment
```
pip install -r requirements.txt
```


### FewNERD
```
bash bash/fewnerd/run_mode.sh [gpu_id] [mode] [N] [K]
    - mode: intra/inter
    - N, K: 5 1, 5 5, 10 1
    e.g., bash bash/fewnerd/run_mode.sh 0 inter 5 1
bash bash/fewnerd/10wat_5shot_mode.sh [gpu_id] [mode]
    - mode: intra/inter
    e.g., bash/fewnerd/10wat_5shot_mode.sh 0 inter
```


# Contact
If there are any questions, please feel free to contact me: Zifeng Cheng (chengzf@smail.nju.edu.cn).
