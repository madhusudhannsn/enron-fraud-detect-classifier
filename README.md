# Enron fraud detect classifier using Decision tree algorithm

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.

## Steps used

1) Used SelectKBest to filter for features with more information.
2) Used train_test_split to split the data into train and test.
3) Used MinMaxScaler to scale the features.
4) Used GridSearchCV with Pipeline having PCA and Decision tree classifier
