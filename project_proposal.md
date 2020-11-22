# Capstone Project Proposal

Chen Liang | For Machine Learning Engineer Nanodegree | 10.04.2020

## Domain Background

This project derived from the observation of my personal devlelopment life which, as a not-experienced data scientist, we usually write bad commit message to our repo which there are tons of updates. These bad commit message, without the review from other team members or senior level person, makes the code base becoming confusing and hard to manage.

[1][2] proposed couple gold rules or patterns of writing a good commit message, given that we have a huge commit message data source **Github,** the aim of the project is to build a naive tool to make help people understand if they are writing a good or poor commit message, it will be helpful if people can correct or rectify their message before they really shoot the commit.

## Problem Statement

For this project, I will be mainly focus on creating a simple classification model which taking the commit message and subject as input, model will output a probabilty indicating if the commit message is a good one. Specifically, combining the knowledge from [2], a good commit message should obey the following rules:

1. Specify the type of commit:
    - feat: The new feature you're adding to a particular application
    - fix: A bug fix
    - style: Feature and updates related to styling
    - refactor: Refactoring a specific section of the codebase
    - test: Everything related to testing
    - docs: Everything related to documentation
    - chore: Regular code maintenance.[ You can also use emojis to represent commit types]
2. Separate the subject from the body with a blank line
3. Limit the subject line to 50 characters
4. Remove unnecessary punctuation marks
5. Do not end the subject line with a period
6. Capitalize the subject line and each paragraph
7. Use the imperative mood in the subject line

We would need to convert the whole commit body to the text or statistical features and train a small neural network model to output the probability.

## Datasets and Inputs

Using given information from [4], we would use the Google Bigquery to fetch the data for commit messages, the dataset has the following schema.

![](imgs/data_schema.jpg)

As mentioned in this table, we will use commiter's name, commit time as Identifier, subject and message as feature. As of the dependent variable, we will use the following rules to create:

- If it has the identifier of the commit message like (fix, update, etc.)
- If the commit subject is over length
- If first not symbol character of commit message is capitalized
- If the commit subject is not end up with period
- If the POS for the first all-alphabet-chars token of the commit message corresponds to a verb

### Overview of the Dataset

Specifically, this is the google big query that we've used, considering the trade-off between cost and capability, we fetched the first 1M records from the commit message table.

In addition to this, considering that some commit message is too small or large, which will hugely impact the size of the dataset file and difficulty that we read, by manually reviewing those super long commit message, they are more likely presenting a technical detail, which I don't think should be seriously considered in this project, so we set 6 and 200 as threshold for filtering the length of the text.

```sql
SELECT committer.name, committer.time_sec, subject, message
FROM `bigquery-public-data.github_repos.commits`
WHERE LENGTH(message) > 6 AND LENGTH(message) <= 200
LIMIT 1000000
```

### Text Features

By checking the commit subject and message body, here are some information about the dataset and feature quality, among 3M records

- There are 0 null-values in the records that we fetched, for both subject and message.
- 10.95% of the subjects are same as the commit messages.
- Picking `\n\n` as delimiter to split the commit message into segments, there are 20.33% messages has multiple segments, and there are 30.65% subjects are identical to the first segment of commit messages.

The length distribution of the subject, full message, and first segment of the message is shown in the following graph

![](imgs/length_distribution.png)

### Label Distribution

According to the previous section, we use the following rules to filter out the good commit message:

- **Identifier**: The "commit identifiers", including `{'implement', 'polish', 'remove', 'refactor', 'add', 'fix', 'rework', 'rename', 'resolve', 'merge', 'update'}` , at least one of them existed in the subject or first segment message
- **Length_ok**: Commit subject is less then 50 characters
- **Not_period_end**: No period in the end of the subject
- **Imperative_mood**: First all-alphabet character token in subject is a verb
- **Capital_first_token**: The first token of commit subject and first segment message is capitalized

If all five rules are satisfied, this commit message would be tagged as a good message, given these rules, the label distribution shows in the belowing table:

![](imgs/label_distribution.png)

Around 1/10 Samples are good messages, identifier rule is the most difficult rule to satisfy, which there are only around 36% samples match the requirement.

## Solution Statement

As proposed in previous section, we are going to convert the commit subject and message to couple statistical or text features, including:

- Rule-based features
    - Length of the string
    - First character is capitalized Flag
    - Last Character is not period flag
- Text features
    - After removing punctuation, strip and lowercase, we would use word embedding to convert the texts to vectors
- Model Training
    - We would try to find ways to integrate these features and feed to a customized neural network

## Benchmark Model

Given that there is not much previous work has been found regarding to solve this issue, I've created a simple `linear_learner` work as binary classifier and treat it as our benchmark. The classifier used the features that described from section `label_distribution`, and treat them all as binary values.

After 15 epochs model training, we've acquired our benchmark model who has reached 100% recall and 83.1% precision for good commit messages. The accuracy of the whole classification task is 97.3%, which is pretty good.

## Evaluation Metrics

We would use the commonly used metrics to evaluate this classification problem, the metrics including precision, recall, f1-score, auc, their definitions are respectively:

**Precision**

[5] Precision is a good measure to determine, when the costs of False Positive is high. For instance, email spam detection. The calculation of precision shows below

![](imgs/precision.png)

**Recall**

[5] Recall actually calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive). Applying the same understanding, we know that Recall shall be the model metric we use to select our best model when there is a high cost associated with False Negative.

![](imgs/recall.png)

**F1-score** 

[5] F1 Score is the harmonic mean of the precision of recall, it is needed when you want to seek a balance between Precision and Recall.

![](imgs/f1_score.png)

In short, given that the dependent variable is generated by human, we believe that there is a chance that human being missed out some part of good commit message. But we expected to see the model can fetch as much good commit message as possible. Thus, if we are facing the trade-off case, we would sacrifice the precision in exchange of recall.

## Project Design

The project development will follows the workflow that shows below:

- Acquire data using google big query, and manage to store in a easy to handle format.
- EDA on data, acquire the statistics which can help us get some insights of the label distribution and data quality, furthermore, to support us adjust the strategy and parameters. like text truncate size, weighting etc.
- Data cleaning, exclude the bad data that we have, like not-English data etc.
- Feature generate, manage to convert the message and subject information to features that feed for neural network.
- Model Training and evaluation.

## Reference

[1] [https://chris.beams.io/posts/git-commit/#separate](https://chris.beams.io/posts/git-commit/#separate)

[2] [https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/](https://www.freecodecamp.org/news/writing-good-commit-messages-a-practical-guide/)

[3] [https://datascience.stackexchange.com/questions/58372/extract-imperative-sentences-from-a-documentenglish-using-nlp-in-python](https://datascience.stackexchange.com/questions/58372/extract-imperative-sentences-from-a-documentenglish-using-nlp-in-python)

[4] [https://github.blog/2017-01-19-github-data-ready-for-you-to-explore-with-bigquery/](https://github.blog/2017-01-19-github-data-ready-for-you-to-explore-with-bigquery/)

[5] [https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9](https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9)