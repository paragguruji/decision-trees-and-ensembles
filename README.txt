=============================================================

Ensemble Classifiers - 1. Decision Tree, 2. Bagging, 
3. Random Forest 4. Adaptive Boosting; and SVM on
the Text Classification task over yelp reviews data
=============================================================

Purdue University Spring 2017 CS 573 Data Mining Homework 4
=============================================================

Author: Parag Guruji
Email: pguruji@purdue.edu
=============================================================
 
Python Version 2.7
=============================================================

Directory structure:

parag_guruji/
	|---__init__.py
	|---hw4.py
	|---report.pdf
	|---README.txt
	|---requirements.txt
	|---output/ (optional)
		|---comparison_by_tss.csv
		|---comparison_by_fcount.csv
		|---comparison_by_depth.csv
		|---comparison_by_tcount.png
		|---comparison_by_tss.png
		|---comparison_by_fcount.png
		|---comparison_by_depth.png
		|---comparison_by_tcount.png
		|---ttest_by_tss.csv
		|---ttest_by_fcount.csv
		|---ttest_by_depth.csv
		|---ttest_by_tcount.csv

=============================================================

usage: hw4.py [-h] [-a analysis] [-t ttest] [-p plot]
              trainingDataFilename testDataFilename modelIdx

CS 573 Data Mining HW4 Ensembles

positional arguments:
  trainingDataFilename  file-path of training set
  testDataFilename      file-path of testing dataset
  modelIdx              Choice of model: 1. Decision Tree 2. Bagging Tree 3.
                        Random Forest 4. Support Vector Machine 5. Ada-
                        Boosting on Decision Tree

optional arguments:
  -h, --help            show this help message and exit
  -a analysis, --analysis analysis
                        Choice of x variable of analysis: 1. TSS = [0.025,
                        0.05, 0.125, 0.25] 2. Feature Count = [200, 500, 1000,
                        1500] 3. Tree Depth = [5, 10, 15, 20] 4. Tree Count =
                        [10, 25, 50, 100] 5. All 4 (default: None)
  -t ttest, --ttest ttest
                        Choice of x variable for ttests - attempts to read the
                        summery stats from output/comparison_by_<lowercase
                        param code>.csv: 1. TSS (tss) = [0.025, 0.05, 0.125,
                        0.25] 2. Feature Count (fcount) = [200, 500, 1000,
                        1500] 3. Tree Depth (depth) = [5, 10, 15, 20] 4. Tree
                        Count (tcount) = [10, 25, 50, 100] 5. All 4 (default:
                        None)
  -p plot, --plot plot  Choice of x variable for drawing plots - attempts to
                        read the summery stats from
                        output/comparison_by_<lowercase param code>.csv: 1.
                        TSS (tss) = [0.025, 0.05, 0.125, 0.25] 2. Feature
                        Count (fcount) = [200, 500, 1000, 1500] 3. Tree Depth
                        (depth) = [5, 10, 15, 20] 4. Tree Count (tcount) =
                        [10, 25, 50, 100] 5. All 4 (default: None)
