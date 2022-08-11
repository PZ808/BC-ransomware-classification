# Binary classification of ransomware bitcoin addresses 

## Introduction

In the last decade, we have witnessed how cryptocurrencies like bitcoin have made their way into malware scams, specifically those which require payments to the hackers for removal. The most well known case is ransomware. Ransomware is rogue compute code that encrypts the victim's filesystem until money is paid to the attacker for decryption. Ransomware operators typically utilize client-server asymmetric key encryption. The public key is used to encrypt the files, while the private key is stored remotely on a server to be accessed when the ransom is paid. Bitcoin is often demanded for ransomware payments, as it provides an anonymous means of transaction on a decentralized peer-to-peer network operating across the globe and without governmental regulation in many countries.
As such, the use of bitcoin poses a challenge to law enforcement agencies targeting ransomware cybercrime. However, the bitcoin network of addresses and transactions is publically available data, and can be scoured for traces and patterns indicating criminal activity. 
Indeed, analysis of the data has revealed that addresses linked to ransomware display certain characteristics. 
For our work, we use a characterization based on [Akcora et al](https://arxiv.org/pdf/1906.07852.pdf), where the network is viewed as a weighted directed graph, to develop an AI models to predict whether a Bitcoin address is being used for ransomware or not.

 ## Data 
 
The dataset we use is [publically available](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset) from the UCI Machine Learning Repository. The data is taken from January 2009 to December 2018. It is a multivariate, time-series table of 2916697 records with 10 attributes per record containing several classes of ransomware. The dataset is non-stationary, has skewed features, and has an extremely unbalanced target variable. 
We simplify our study by ignoring the time dependence. We also convert the labels to binary form according to whether an address is ransomware or not.
  
 
 ### Description of features
---
Our definition of features follows [Akcora et al](https://arxiv.org/pdf/1906.07852.pdf), and derives from a weighted directed graph description of the Bitcoin network. The data is split into 24hr intervals, where in each time interval certain **"starter transactions"** are identified. These are those which are spent outputs from the previous time window, and provide the coins for the addresses in the current time window.

Given a set of starter transactions, [Akcora et al](https://arxiv.org/pdf/1906.07852.pdf) define the features
-  **address** is a public address (key) which can receive and send bitcoins        
-  **income** of an address $a$ is the total amount of coins output to it measured in Satoshis (1 millionth of a BC)
-  **neighbors** of an address $a$  is the number of transactions which have $a$ as one of its output addresses
-  **weight** of an address is the sum of the fraction of *coins* that come from a starter transaction and merge at the address.
-  **length** of an address $a$ is the number of non-starter transactions connected to $a$ on the longest chain.  A length of zero implies that the address is the recipient of starter transaction.
-  **count** of an address $a$ is the number of starter transactions which are connected to $a$ in a chain
- **looped** is the number of starter transactions connected to $a$ by more than one chain. 
 
 ### Workflow
 
 #### 1. Reading, exploring, and processing the data.
 The notebook bc_rans_calc.ipynb (bc_rans-v2.ipynb) is for processing the data. Its jobs include
 - reading the csv
 - test/train split
 - binary encoding the target variable (1=ransomware, 0=white), leaving approximately 98.5% of the labels in the negative class
 - plotting PDFs of the variables
 - variable transformations reducing skew and improving Gaussianity
 - introducing newly engineered features
 - encoding categorical variables and stardardizing numerical features
 - outputing data to pickle binaries with original data, derived data, encoded data
 
#### 2. Resampling and Modeling 
The notebook bc_rans-readpickle-v2.ipynb is to resample and run several binary classification models optimized for recall. Its jobs include
- reading the X,y from pickled files  
- upsampling with SMOTE, random down resampling, and recombining data 
- univariate analysis: PDF comparisons of each feature for the positive and negative class
- multivariate anaylsis: heatmap with correlations between features
- hyperparameter tuning (optimized for recall) with cross validation for a family stochastic gradient descent, random forest, and extreme gradient boosted decision tree models
- feature importance ratings
- scoring the models: generating confusion matrices, ROC/AUC figs, etc.
