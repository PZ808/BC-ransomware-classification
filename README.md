# IH_FINAL Project -  Binary classification of ransomware bitcoin addresses 

## Introduction

In the last decade, we have witnessed how cryptocurrencies like bitcoin have made their way into malware scams, specifically those which require payments to the hackers for removal. The most well known case is ransomware. Ransomware is rogue compute code that encrypts the victim's filesystem until money is paid to the attacker for decryption. Ransomware operators typically utilize client-server asymmetric key encryption. The public key is used to encrypt the files, while the private key is stored remotely on a server to be accessed when the ransom is paid. Bitcoin is often demanded for ransomware payments, as it provides an anonymous means of transaction on a decentralized peer-to-peer network operating across the globe and without governmental regulation in many countries.
As such, the use of bitcoin poses a challenge to law enforcement agencies targeting ransomware cybercrime. However, the bitcoin network of addresses and transactions is publically available data, and can be scoured for traces and patterns indicating criminal activity. 
Indeed, analysis of the data has revealed that addresses linked to ransomware display certain characteristics. 
For our work, we use a characterization based on [Akcora et al](https://arxiv.org/pdf/1906.07852.pdf), where the network is viewed as a weighted directed graph, to develop an AI models to predict whether a Bitcoin address is being used for ransomware or not.


The dataset we use is [publically available](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset) at UCI Machine Learning Repository. The data is from 2009 January to 2018 December. It has is a multivariate, time-series containing 2916697 records with 10 attributes per record. In our study, we ignore the time dependence and simplify the classification by converting the labels to binary. 


 ## Data 
 
 ### Description of features
---
Our definition of features follows [Akcora et al](https://arxiv.org/pdf/1906.07852.pdf), and derives from a weighted directed graph description of the Bitcoin network. The data is split into 24hr intervals, where in each time interval certain **"starter transactions"** are identified. These are those which are spent outputs from the previous time window, and provide the coins for the addresses in the current time window.

Given a set of starter transactions, [Akcora et al](https://arxiv.org/pdf/1906.07852.pdf) define the features
- The **address** is a public address (key) which can receive and send bitcoins.        
- The **income** of an address $a$ is the total amount of coins output to it measured in Satoshis (1 millionth of a BC)
- The number of (in) **neighbors** of an address $a$  is the number of transactions which
have $a$ as one of its output addresses:
- The **weight** of an address is the sum of the fraction of *coins* that come from a starter transaction and merge at the address. Weight quantifies the merge behavior (i.e., the transaction
has more input addresses than output addresses), where coins
in multiple addresses are each passed through a succession of
merging transactions and accumulated in a final address.
- The **length** of an address $a$ is the number of non-starter transactions connected to $a$ on the longest chain.  A length of zero implies that the address is the recipient of starter transaction.
address is an output address of a starter transaction.
- The **count** of an address $a$ is the number of starter transactions
which are connected to $a$ in a chain, where a chain is
defined as an acyclic directed path originating from any starter
transaction and ending at address $a$. he count feature represents
information on the number of transactions, whereas the weight
feature represents information on the amount (what percent of
these transactions’ output?) of transactions.
- **looped** is the number of starter transactions connected to $a$ by more than one path (chain). 
 
 ### Workflow
 
 
 
 
