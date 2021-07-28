# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
### Problem
- This project uses a Bank Marketing Dataset from the [USI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
- The dataset conatins personal details about clients such as age, job, marital status, education, etc among other attributes. 
- This is a **_classification_** (2 class-classification) problem with the goal to predict whether or not a client will subscribe to a term deposit with the bank. 
- The data is classified using the column label y in the dataset that contains binary values ('yes' and 'no').

### Project Workflow Steps
![Image of Pipeline Architecture](images/pipeline_architecture.png)

### Solution Aproach
- This project used two approaches to find the best possible model for classifying the given dataset:
  1. Scikit-Learn based logistic regression using the HyperDrive for hyperparameter tuning
  1. Automated Machine Learning was used to build and choose the best model
- Both these approaches were extecuted using _Jupyter Notebook and the Azure ML SDK_.
 
### Results
* The best performing model was a **_VotingEnsemble_** algorithm that was selected through AutoML with an accuracy of **0.9159**.
* The Logistic Regression model whose hyperparameters were tuned using HyperDrive gave an accuracy of **0.9102**.

## Approaches
- Two approaches were used in this project to classify the given data and come up with the best possible model:
  1. A Scikit-Learn Pipeline Regression
  1. Automated ML (AutoML)
- Both these approaches were extecuted using _Jupyter Notebook and the Azure ML SDK_.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
