# stBERT
BERT for spatial transcriptomics

## Dependencies
stBERT is implemented in the pytorch framework (tested on Ubuntu 20.04). 
~~~
python=3.8
r-base=3.6.3
pytorch=1.12.1
cudatoolkit=11.3
biopython=1.79
mkl=2024.0
scikit-misc=0.2.0
louvain=0.8.2
~~~

## Datasets

All datasets used in our paper can be found in:

* DPLFC: 
  The primary source: https://github.com/LieberInstitute/spatialLIBD; 
  The processed version: https://www.nature.com/articles/s41593-020-00787-0.

* Human breast cancer: 
  The primary source: https://www.10xgenomics.com/resources/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0; 
  The processed version: https://github.com/JinmiaoChenLab/SEDR_analyses/.

* Slide-seqV2 mouse olfactory bulb:  
  The primary source: https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary; 
  The processed version: https://stagate.readthedocs.io/en/latest/T3_Slide-seqV2.html.

* Stereo-seq mouse olfactory bulb: 
  The processed version: https://github.com/JinmiaoChenLab/SEDR_analyses/.

* MOSTA database: 
  The primary source: https://db.cngb.org/stomics/mosta/; 
  The processed version: https://db.cngb.org/stomics/mosta/download/.

## pre-train and fine-tune
set the data name in cfg.py, and then run main.py

The development of this code was partly facilitated by [stAA](https://github.com/CSUBioGroup/stAA), thanks!

