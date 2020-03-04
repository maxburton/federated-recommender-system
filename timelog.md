# Timelog

* An Aggregator For Recommendation Engines
* Max Kirker Burton
* 2260452B
* Nikos Ntarmos

## Guidance

* This file contains the time log for your project. It will be submitted along with your final dissertation.
* **YOU MUST KEEP THIS UP TO DATE AND UNDER VERSION CONTROL.**
* This timelog should be filled out honestly, regularly (daily) and accurately. It is for *your* benefit.
* Follow the structure provided, grouping time by weeks.  Quantise time to the half hour.

## Week 0

### 23 Sept 2019

* *1 hour* Meeting with supervisor
* *1 hour* Created Github repository and create initial issues, shared link with supervisor

## 24 Sept 2019

* *2 hours* Read through mark scheme and project guide documents
* *4 hours* Introductory lectures
* *1 hour* Setting up template files and writing summer report

## 27 Sept 2019
* *1 hour* Researching existing recommender engines built in Python

## 28 Sept 2019
* *1 hour* Continued researching existing recommender engines built in Python

## Week 1

## 01 Oct 2019
* *1 hour* Set up latex and dissertation template in sublime text ready to build

## 3 Oct 2019
* *2 hours* Discussion with Nikos
* *1 hour* Background reading on python recommender libraries

## Week 2

## 7 Oct 2019
* *3 hours* Configuring SurPRISE library

## 8 Oct 2019
* *2 hours* Practice using SurPRISE library with movielens dataset

## 9 Oct 2019
* *2 hours* Reading through the chapters Nikos recommended for me via email

## 10 Oct 2019
* *1 hour* Reading through the chapters Nikos recommended for me via email

## Week 3

## 12 Oct 2019
* *1 hour* Researching Item2Vec

## 13 Oct 2019
* *2 hours* Installing and experimenting with light.fm (spotlight had compatibility errors)

## 15 Oct 2019
* *1 hour* Reading Nikos' email about different algorithm mappings

## 16 Oct 2019
* *2 hours* Experimenting with light.fm

## 17 Oct 2019
* *1 hour* Meeting with Nikos

## Week 4

## 21 Oct 2019
* *2 hours* Researching spotlight

## 24 Oct 2019
* *2 hours* Researching metrics/evaluation and how to order dataset

## Week 5

## 31 Oct 2019
* *1 hour* Meeting with Nikos

## 02 Nov 2019
* *2 hours* Research about how to slice data and created a plan for tomorrow

## 03 Nov 2019
* *4 hours* Implemented data slicing functions
* *2 hours* Implemented testing and added some unit tests
* *1 hour* Implemeneted logging

## Week 6

## 04 Nov 2019
* *6 hours* Added tests for each data slicing function, and added/fixed some of them

## 05 Nov 2019
* *5 hours* Continued with data handling functions
* *2 hours* Started creating the main algorithm handler class

## 06 Nov 2019
* *1 hour* Meeting with Nikos

## Week 7

## 11 Nov 2019
* *2 hours* Reading through Richard's thesis, specifically to understand multi-engine mappings and how to implement them

## 13 Nov 2019
* *1 hour* Meeting with Nikos

## Week 8

## 18 Nov 2019
* *2 hours* Researching item-item and user-item collaborative filtering preparing to implement

## 19 Nov 2019
* *3 hours* More work on data slicing with added functionality
* *4 hours* Implemented knn item-based collaborative filtering algorithm

## 21 Nov 2019
* *5 hours* Fixing pivot method

## 22 Nov 2019
* *3 hours* Fixing pivot method
* *4 hoours* Fixing sparse matrix not indexing correctly

## Week 9

## 25 Nov 2019
* *2 hours* Fixed pivot method and tidied up code

## Week 10

## 01 Dec 2019
* *4 hours* Adding tests for knn

## 04 Dec 2019

* *1 hour* meeting with Nikos

## Week 11

## 13 Dec 2019
* *2 hours* Adding tests for knn and researching other algorithms
* *1 hour* researched user-item collaborative filtering technique

## Week 12

## 17 Dec 2019
* *4 hours* Implemented first draft federator (working on knn on split datasets) and modified data_handler to handle new datasets

## 18 Dec 2019
* *1 hour* Meeting with Nikos

## Week 13

## 23 Dec 2019
* *2 hours* Fixed Spotlight example and implemented a simple spotlight knn algorithm

## Week 14

## 30 Dec 2019
* *3 hours* Improved Spotlight alg and made it fit with existing data handling functions

## Week 16

## 17 Jan 2020
* *3 hours* Implemented a basic LightFM algorithm with WARP model

## 18 Jan 2020
* *3 hours* Added a BPR model to the LightFM model

## Week 17

## 20 Jan 2020
* *4 hours* Improved LightFM alg to make it fit with simple federator
* *3 hours* Updated my knn alg to accomodate users by comparing all their ratings > 4.0 (or a certain threshold) and merging results

## 22 Jan 2020
* *4 hours* simple_federator now works with user ids
* *1 hour* scores returned for lightfm warp and bpr algs

## 24 Jan 2020
* *1 hour* Meeting with Nikos and Richard, discusses next steps and how to map scores from different algs

## 26 Jan 2020
* *3 hours* Created class that handles golden lists from algs
* *1 hour* Created basic federator class to house all important variables

## Week 18

## 29 Jan 2020
* *5 hours* continued work on basic federator class, and started work on a simple score mapper

## Week 19

## 3 Feb 2020
* *2 hours* implemented running algs on individual splits and comparing that to golden lists (knn) (e.g. imagine them as movie providers netflix, amazon, etc) (next step: see if federating these provides better scores)
* *1 hour* ditto with lfm (started)

## 6 Feb 2020
* *2 hours* continued work on lfm, progressing on making lfm work with non standard datasets (e.g. slices)

## 8 Feb 2020
* *1 hour* continued debugging lfm to work with splits

## 9 Feb 2020
* *6 hours* lfm now works on splits, completed individual splits class

## Week 20

## 12 Feb 2020
* *4 hours* added basic alg mapper, that normalises, sorts and trims data from a split in knn and lfm.

## 13 Feb 2020
* *1 hours* meeting with Nikos

## Week 21

## 18 Feb 2020
* *4 hours* Implemented surprise svd alg
* *2 hours* Implemented machine learned mapper

## 19 Feb 2020
* *4 hours* Implemented federator w/ machine learned mapping
* *1 hour* Meeting with nikos

## 21 Feb 2020
* *3 hours* Started writing diss

## Week 22
 
## 24 Feb 2020
* *2 hours* Continued writing diss (related work)
* *2 hours* Implemented sklearn's ndcg score

## 25 Feb 2020
* *2 hours* Continued writing diss (related work)

## 27 Feb 2020
* *1 hour* meeting with nikos
* *2 hours* fixing ndcg scores giving key errors

## Week 23

## 2 Mar 2020
* *2 hours* Continued writing diss (abstract, background)

## 4 Mar 2020
* *2 hours* Fixing ndcg score function