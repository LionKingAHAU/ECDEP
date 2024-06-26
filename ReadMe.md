
# 0. Prerequisite 

  

> If you want to use the ECDEP model, you might need to be equipped with the env below：

  

|Package|Version|
|---|---|
|Python|>=3.8|
|Numpy|>=1.19.4|
|Pandas|>=1.0.5|
|Tensorflow|>=2.4.0|
|Sklearn|>=0.23.1|
|Tiles|>=1.0.4|

  

# 1. Brief Introduction 

  

ECDEP is a deep learning model designed for identifying essential proteins. This model takes three types of biological data as input: Protein-Protein Interaction (PPI) networks, gene expression profiles, and subcellular localization information.

  

The model's workflow involves several steps:
1. First, it constructs a dynamic PPI network using both the PPI network and gene expression data. 
2. Next, it applies the TILES algorithm to identify communities that emerge during the evolutionary process. 
3. Then, These communities are then prioritized using the SVM-RFE (Support Vector Machine - Recursive Feature Elimination) algorithm.
4. In the final step, the selected communities, along with subcellular localization data, are combined to create a feature representation for each protein. This feature representation is then processed through dense layers in the model to generate predictions regarding protein essentiality.

  

# 2. How to Use ECDEP

  

> Our project is comprised of five main files.
> Please make sure that the **data** file and the **ECDEP** file are in the same directory. 
> Then, **enter the ECDEP directory and Run the commands as follows step by step**.
> We have provided a demo of yeast(Krogan) in the Dynamic Network Demo directory, you can directly run the scripts.

  

1. DPPIN.py: This script helps to build dynamic PPIN.

  

```python

python DPPIN-step1.py

```

  

2. streaming-step2.py: This script helps to change DPPIN into an interaction streaming source.

```python

python streaming-step2.py

```

3. getEtiles-step3.py: This script helps to use TILES to generate communities.

```python

python getEtiles-step3.py

```

4. prioritization-step4.py: This script helps to use SVM-RFE to select informative communities.

```python

python prioritization-step4.py

```

5. ECDEP-step5.py: this script uses a deep learning model to predict essential proteins.

```python

python ECDEP-step5.py

```  


# 3. Available Data and Test

  

All datasets used in our paper and test data are stored in [zenodo](https://zenodo.org/records/8363124).

  

> In the `data\Dynamic Network Demo` dir, we provide the data of Krogan for a demo, you can directly run the script with it.


> In the `data\Data For Test` dir, we provide other species data used in our paper. We have retrieved the community features. 
  
# 4. Citing Information

If you find our work useful, please cite Ye, C., Wu, Q., Chen, S. et al. ECDEP: identifying essential proteins based on evolutionary community discovery and subcellular localization. BMC Genomics 25, 117 (2024). https://doi.org/10.1186/s12864-024-10019-5
