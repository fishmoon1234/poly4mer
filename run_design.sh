# tunable hyperparameters
c_tig=0.1
c_qpua=1
c_sea=1
c_co=1
#idx=22
max_epochs=500
#lr=5e-1

echo -e "PEFORMING OPTIMIZATION.\n"

# python fire_optimize_smiles.py \
#     --lr_start ${lr} \
#     --max_epochs ${max_epochs} \
#     --c_tig ${c_tig} \
#     --c_qpua ${c_qpua} \
#     --c_sea ${c_sea} \
#     --c_co ${c_co} \
#     --smiles_idx ${idx} \

#test multiple initial polymers
for idx in {11,17};
do
#test with multiple learning rates
    for lr in {0.5,1.0};
    do
        echo "Running idx=$idx with lr=$lr ..."
        python fire_optimize_smiles.py \
        --lr_start ${lr} \
        --max_epochs ${max_epochs} \
        --c_tig ${c_tig} \
        --c_qpua ${c_qpua} \
        --c_sea ${c_sea} \
        --c_co ${c_co} \
        --smiles_idx ${idx} 

    done
done

wait
echo 'ALL runs finished'

