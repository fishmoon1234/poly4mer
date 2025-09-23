# poly4mer
A multi-physics synthesis based chemical language model for polymer fire property prediction and inverse design.

![Poly4mer architecture.](https://github.com/fishmoon1234/poly4mer/blob/main/poly4mer.png)

This repository houses the code for our work in forward prediction and inverse polymer design using chemical language models.

**Highlights**:

1. The model is developed based on ![SMI-TED](https://github.com/IBM/materials/tree/main), and trained to predict 4 flammability metrics: time to ignition (Tig), peak heat release rate (pHRR), smoke extinction area (SEA), and carbon monoxide yield (CO) from polymer SMILES.

2. We introduce a principled multi-physics synthesis framework to data-scarce learning by embedding domain knowledge directly into language model training, enabling robust generalization beyond pure data-driven chemical language models.

3. We design a new strategy for introducing new tokens into pretrained language models via encoder-decoder architecture expansion, leveraging monomer representations to capture complex polymer semantics.

4. We architect an autoencoding system coupling predictive modeling with generative design, enabling inverse polymer design via latent-space exploration and structure reconstruction for targeted applications.

## Requirements
<!-- - [PyTorch](https://pytorch.org/) -->
- Python 3.10.18
- numpy==1.23.5
- torch==2.5.1
- transformers==4.53.2
- tokenizers==0.21.2
- rdkit==2025.3.3
- pandas==1.4.0

## Pretrained models
Please download and extract pretrained models at
- [Our decoder and predictor models](https://drive.google.com/drive/folders/1nvXUwWs3lRk7Xsa6OuBFqlWmfSSQ_Ds4?usp=sharing)
- [The pre-trained encoder model in smi-ted](https://github.com/IBM/materials/tree/main/models/smi_ted/smi_ted_light)


## Running experiments

**Property predictor**:

1. Add the pSMILES of polymers to be predicted in 
```
example_smiles.txt
```

2. Edit the corresponding polymer index in 
```
run_property_prediction.sh
```

3. Run predictor with the shell script
```
./run_property_prediction.sh
```

**Polymer design**:

1. Add the pSMILES of candidate initialization polymers in 
```
example_smiles.txt
```

2. Edit the corresponding polymer index in 
```
run_design.sh
```
together with the penality parameters and optimization hyperparameters.

For the penality parameters, one can tune
```
c_tig=0.1
c_qpua=1
c_sea=1
c_co=1
```
to change weights between different fire properties.

For the optimization hyperparameters, one can change the maximum optimization steps and the step size.

3. Run inverse design with the shell script
```
./run_design.sh

```

4. Outputs are stored in
```
results/optimized_smiles_canonicalized/tig${c_tig}_phrr${c_qpua}_sea${c_sea}_co${c_co}/optimized_smiles_canonicalized_${idx}_lr_${lr}.txt
```

## Citation

