# Udacity-ML-Engineer-Capstone-Project
A classification model to identify a good commit message

This proejct is mainly focus on creating a simple classification model which taking the commit message and subject as input, model will output a probabilty indicating if the commit message is a good one.

## What's in the box
There are three main stuff in this repo:
- folder `codes` include the `train.py`, `predict.py` and `model.py` which corresponding to the sagemaker training job, sagemaker inference endpoint to host the model, and the model architecture.
- `data_eda_and_baseline_model.ipynb` which acquires the data from google bigquery, making exploration to the dataset, create the rule-based system and simple binary classifier as the benchmark model.
- `ml_pipeline.ipynb` which is the main pipe for creating the neural network model used for this commit message classification task. Including feature engineering and train/deploy the sagemaker model.

## Replicate the result
You may need the following stuff to replicate the result
- Corresponding packages shows in the notebook.
- A google bigquery credential file to allow you make the query from notebook.

Once you have them, simply run `data_eda_and_baseline_model.ipynb` and then `ml_pipeline.ipynb`, I expect that you may acuquire a similar experiment result as I did, the result may be slightly different given that the database is keep updating new commit message.

## Reference
For more information, please reference to `project_proposal.md` and `project_report.md`
