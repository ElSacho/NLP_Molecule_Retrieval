# NLP_Molecule_Retrieval

## Overview
This project lies at the intersection of Natural Language Processing (NLP) and cheminformatics, aiming to bridge the gap between textual descriptions and molecular structures represented as graphs. By leveraging advanced machine learning techniques, specifically contrastive learning, we developed models capable of accurately matching molecules to their corresponding textual descriptions. Our approach involves co-training a text encoder and a molecule encoder to align similar text-molecule pairs in a shared representation space, achieving significant precision improvements with an LRAP score of over 0.94 on the test dataset.

## Introduction
The challenge of accurately matching molecular structures with textual descriptions presents a unique intersection of NLP and molecular science. Our project explores this novel area by developing methods that employ contrastive learning to effectively co-train text and molecule encoders, thus facilitating the identification of corresponding molecules for given text queries. This is achieved despite the absence of direct textual information about the molecules, highlighting the potential of our models to navigate the complex relationship between language and molecular graphs. Our models' effectiveness is quantified through the Label Ranking Average Precision (LRAP) score, demonstrating their capability to achieve high accuracy in molecule retrieval.

## Key Features
- **Contrastive Learning:** Utilizes contrastive loss, triplet loss, and Lifted Structured Loss to learn effective representations by contrasting similar and dissimilar pairs of data points.
- **Boosting Strategies:** Incorporates methods inspired by boosting and a genetic algorithm-inspired strategy to enhance model diversity and performance.
- **High LRAP Scores:** Achieves an LRAP score of 0.90 with our initial approach and over 0.94 with our refined boosting strategies on the test dataset.
- **Comprehensive Model Comparison:** Conducts extensive experiments with over 70 different hyperparameters to identify the most impactful configurations.

## Getting Started
To run the code, follow these steps:
1. **Install Requirements:** Install the necessary requirements by running `pip install -r requirements.txt`.
2. **Prepare the Dataset:** Separate the validation dataset similarly to the test dataset.
3. **Configuration:** Choose a configuration file and run the model by executing `python trainer.py path_to_your_conf_file`. Make sure to set the experiment name, whether to load an existing model, and the save names for your models and CSV files in the config file.

## Contribution
For further details on each specific parameter and to access the configuration files, visit our [GitHub repository](https://github.com/ElSacho/NLP_Molecule_Retrieval). All results can be reviewed in the corresponding files within the *log* folder on GitHub, analyzed using TensorBoard.


