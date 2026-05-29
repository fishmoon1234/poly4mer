# tunable hyperparameters
c_tig=0.1
c_qpua=1
c_sea=1
c_co=1
#idx=22
max_epochs=500
flux=75
thickness=6
#lr=5e-1

echo -e "PERFORMING OPTIMIZATION.\n"

# python fire_optimize_smiles.py \
#     --lr_start ${lr} \
#     --max_epochs ${max_epochs} \
#     --c_tig ${c_tig} \
#     --c_qpua ${c_qpua} \
#     --c_sea ${c_sea} \
#     --c_co ${c_co} \
#     --smiles_idx ${idx} \

#test multiple initial polymers
for idx in {1..28};
do
#test with multiple learning rates
    for lr in {0.005,0.01,0.02,0.05,0.1};
    do
        echo "Running idx=$idx with lr=$lr ..."
        python fire_optimize_smiles_v2.py \
        --lr_start ${lr} \
        --max_epochs ${max_epochs} \
        --c_tig ${c_tig} \
        --c_qpua ${c_qpua} \
        --c_sea ${c_sea} \
        --c_co ${c_co} \
        --smiles_idx ${idx} \
        --flux ${flux} \
        --thickness ${thickness}

    done
done

wait
echo 'ALL runs finished'

