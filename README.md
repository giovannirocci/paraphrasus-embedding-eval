(DRAFT - TO BE UPDATED FOR BETTER USABILITY)
# Reproduction and further work code repository for [PARAPHRASUS: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models]

This repository contains the code, configuration files, datasets, and scripts to reproduce the experiments and results from the preprint "PARAPHRASUS: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models" . 

This github repository provides the necessary tools to reproduce/reanalyse and further experiment with new models and configurations.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Further Experimentation](#further-experimentation)
- [BibTeX Reference](#bibtex-reference)
- [Datasets and License](#datasets-and-license)
- [Further Support](#further-support)

## Overview

This repository allows replication of the experiments from the research titled "PARAPHRASUS: A Comprehensive Benchmark for Evaluating Paraphrase Detection Models" and is extendable to allow further experimentation, qualitative analysis. It includes:

- Original predictions generated using the models described in the paper, useful for further qualitative or quantitative analysis.
- Scripts and configuration files to reproduce the results.
- Utility scripts to reproduce statistics plots and so on.

## Directory Structure

The repository is organized as follows:

\`\`\`
├── configs
│   └── Various configuration files for the models and experiments.
├── datasets
│   └── Contains the datasets used in the experiments, along with utility scripts.
├── models
│   └── Empty models file used in the experiments.
├── extract_results_*.py
│   └── Scripts for extracting and analyzing results from experiments.
├── full_experimentation_pipeline.py
│   └── A pipeline script to run a new experiment.
├── mass_experiment_with_different_configs.sh
│   └── A bash script to run multiple experiments with different configurations.
\`\`\`

## Reproducing the Experiments

Scripts are available to extract and reproduce the original results from the paper. You can use the \`extract_results_*.py\` scripts for different datasets and models, or run the entire experimentation pipeline with the \`full_experimentation_pipeline.py\` script by passing a config file parameter.

## Further Experimentation

You can run your own experiments using different configurations provided in the \`configs\` folder. For information about hte different configurations, please check the \`full_experimentation_pipeline.py\` file

## BibTeX Reference

If you use this repository or the reproduced results in your work, please cite the original paper:

\`\`\`bibtex
@misc{michail2024paraphrasuscomprehensivebenchmark,
      title={PARAPHRASUS : A Comprehensive Benchmark for Evaluating Paraphrase Detection Models}, 
      author={Andrianos Michail and Simon Clematide and Juri Opitz},
      year={2024},
      eprint={2409.12060},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.12060}, 
}
\`\`\`

## Datasets and License

This repository inherits its license from the original release, and all datasets used are publicly available from the following links (among many others):

1. **PAWS-X**  
   Link: [PAWS-X Dataset](https://github.com/google-research-datasets/paws/tree/master/pawsx)

2. **SICK-R**  
   Link: [SICK-R Dataset](https://zenodo.org/records/2787612)

3. **MSRPC**  
   Link: [Microsoft Research Paraphrase Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398)

4. **XNLI**  
   Link: [XNLI Dataset](https://cims.nyu.edu/~sbowman/xnli/)

5. **ANLI**  
   Link: [Adversarial NLI (ANLI)](https://github.com/facebookresearch/anli) 

6. **Stanford NLI (SNLI)**  
   Link: [SNLI Dataset](https://nlp.stanford.edu/projects/snli/)

7. **STS Benchmark**  
   Link: [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)

8. **OneStopEnglish Corpus**  
   Link: [OneStopEnglish Corpus](https://github.com/nishkalavallabhi/OneStopEnglish)


Within this work, we introduce a dataset (and an annotation on an existing one) which are also available within our repository under the same license as the source dataset

1. **AMR Paraphrases** Link: (https://github.com/amrisi/amr-guidelines)

2. **STS Benchmark (STS-H) with Human Annotation - Consensus (Column)** 
   Link: [STS Benchmark](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark)


## Further Support

In the future, we will work towards adding more datasets (also multilingual) and to make the benchmark more compute efficient. We plan to enable easier experimentation for our benchmark. If you are interested in contributing or need support reproducing/recreating/extending the results, please reach out to andrianos.michail@uzh.ch
