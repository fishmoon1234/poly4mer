echo -e "PERFORMING PROPERTY PREDICTION.\n"

flux=75
thickness=6

for idx in {11,12};
do
    echo "Running idx=${idx} ..."
    python fire_property_predict_v2.py \
        --smiles_idx ${idx} \
        --flux ${flux} \
        --thickness ${thickness}
done

wait
echo 'ALL RUNS FINISHED.'
