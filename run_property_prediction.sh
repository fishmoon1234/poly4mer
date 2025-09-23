echo -e "PEFORMING PROPERTY PREDICTION.\n"

for idx in {11,12};
do
    echo "Running idx=${idx} ..."
    python fire_property_predict.py --smiles_idx ${idx} 
done

wait
echo 'ALL RUNS FINISHED.'

