# poly4mer
A multi-physics synthesis based chemical language model for polymer fire property prediction and inverse design.

![Poly4mer architecture.](https://github.com/fishmoon1234/poly4mer/blob/main/poly4mer.png)

This repository houses the code for our work in forward prediction and inverse polymer design using chemical language models.

**Highlights**:

1. The model is developed based on [SMI-TED](https://github.com/IBM/materials/tree/main), and trained to predict 4 flammability metrics: time to ignition (Tig), peak heat release rate (pHRR), smoke extinction area (SEA), and carbon monoxide yield (CO) from polymer SMILES.

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
- [Our asterisk encoder, decoder module and fire property predictors](https://drive.google.com/file/d/1ZnEITpI7JgDU3CAK1YLbguGDrsh0FqA6/view?usp=drive_link)
- [The pre-trained encoder model in smi-ted](https://drive.google.com/file/d/1nX7ipiXXYR0xEHOZOjEhi75qMW2Ryw5J/view?usp=drive_link)

After you extracted these models in the root folder of poly4mer, the folder should look like
```
poly4mer/
|-- smi_ted_light/
    |-- bert_vocab_curated.txt
    |-- fast_transformers/
    |-- load.py
    |-- smiles_prediction_results.csv
    |-- smi-ted-Light_40.pt
    ...
|-- checkpoint/
    |-- decoder_predictor/
    |-- optimal_model/
    |-- Star_Lang_1/
    ...
|-- bert_vocab_curated.txt
|-- fire_optimize_smiles.py
|-- Canonicalize_for_Optimize.py
|-- args_nl.py
|-- utils.py
|-- models.py
|-- fire_property_predict.py
|-- example_smiles.txt
|-- run_property_prediction.sh
|-- run_design.sh
...

```

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

4. Candidate polymers are stored in
```
results/optimized_smiles_canonicalized/tig${c_tig}_phrr${c_qpua}_sea${c_sea}_co${c_co}/optimized_smiles_canonicalized_${idx}_lr_${lr}.txt
```
The optimization path and estimated fire property values can be found in 
```
results/log_optimized_prop/tig${c_tig}_phrr${c_qpua}_sea${c_sea}_co${c_co}/log_optimized_props_${idx}_lr_${lr}.txt
```

## Citation

If you find our models useful, please consider citing our papers:

```
@article{liu2025harnessing,
  title={Harnessing large language models for data-scarce learning of polymer properties},
  author={Liu, Ning and Jafarzadeh, Siavash and Lattimer, Brian Y and Ni, Shuna and Lua, Jim and Yu, Yue},
  journal={Nature Computational Science},
  volume={5},
  number={3},
  pages={245--254},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
@inproceedings{yin2025fake,
  title={Fake It Till You Make It: Multi-Physics Synthesis Breaks the Data Barrier in Chemical Language Models},
  author={Yin, Naiyu and Liu, Ning and Chen, Jiuzhou and Lattimer, Brian Y and Lua, Jim and Yu, Yue},
  booktitle={NeurIPS 2025 Machine Learning and the Physical Sciences Workshop}
}
```

