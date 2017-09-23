# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:38:27 2017

@author: Parag
"""

import sys
import argparse
import numpy as np
from codecs import open as codecs_open
from re import sub as re_sub
from operator import itemgetter
from random import sample, shuffle, choice as choose_wr
from sys import stdout
from time import time
from os import path, makedirs
import csv
from matplotlib import pyplot
from collections import Counter
from scipy.stats import ttest_ind_from_stats
import math
# from pprint import pprint

MODELS = ["DT", "BT", "RF", "AB", "SVM"]
ALPHA_TTEST = 0.05
LAMBDA = 0.01
ETA_SVM = 0.5
K = 10
DEFAULT_TRAINING_CYCLES = 100
DEFAULT_STOPWORD_COUNT = 100
DEFAULT_FEATURE_COUNT = 1000
DEFAULT_TREE_COUNT = 50
DEFAULT_TREE_DEPTH = 10
DEFAULT_TSS = 0.25
GIVEN_VALUES = {'TSS': [0.025, 0.05, 0.125, 0.25],
                'FCOUNT': [200, 500, 1000, 1500],
                'DEPTH': [5, 10, 15, 20],
                'TCOUNT': [10, 25, 50, 100]}


def timing(f):
    """Timer Decorator. Unused in final submission to avoid unwanted printing.
        paste @timing on the line above the def of function to be timed.
    """
    def wrap(*args, **kwargs):
        time1 = time()
        ret = f(*args, **kwargs)
        time2 = time()
        print '%s function took %0.3f ms' % \
              (f.func_name, (time2 - time1) * 1000.0)
        sys.stdout.flush()
        return ret
    return wrap


def euclidean(a, b):
    """Returns Euclidean distance between 2 vectors a and b
    """
    return np.sqrt(sum([(ai - bi)**2 for ai, bi in zip(a, b)]))


def get_class_labels(model):
    """Return appropriate class labels for given model
    """
    if model in ["DT", "BT", "RF"]:
        return {"positive": 1, "negative": 0}
    if model == "SVM":
        return {"positive": 1, "negative": -1}


def read_tsv_file(file_name):
    """reads the given tab separated data file. If a line has only review text\
        (missing review_id and/or class_label, then appends it to the text of \
        previous valid review)
    """
    data = {}
    last_id = -1
    infp = codecs_open(file_name, "r", encoding='utf-8')
    for line in infp:
        line = [l.strip() for l in line.split('\t')]
        if len(line) != 3:
            if last_id >= 0:
                data[last_id]['review'] += " " + line[0].strip()
            else:
                raise ValueError
        else:
            data[int(line[0])] = {'id': int(line[0]),
                                  'class': int(line[1]),
                                  'review': line[2]}
            last_id = int(line[0])
    return data


def preprocess(data_set):
    """Preprocesses each review text in given dataset as following:
        1. Convert everything to lowercase
        2. Remove all characters except alphanumeric characters and whitespaces
        3. Split words on whitespaces
        4. Keep unique words of review in a set, add it to the record dict as \
            a value with key 'bow', discard original text entry from record
    """
    for i in data_set.keys():
        data_set[i]['bow'] = \
            Counter(re_sub(r'[^\w'+' '+']',
                           '',
                           data_set[i].pop('review').lower()).split())
    return data_set


def build_ordered_vocabulary(preprocessed_training_set):
    """Returns a list of unique words sorted in descending order of their \
        frequency. (frequency: no. of records containing that word)
        Tie Resolution: words with equal frequency are randomly shuffled \
            within the range of ranks their respective frequency had.
    """
    vocab = {}
    ordered_vocab = []
    for i in preprocessed_training_set:
        for w in preprocessed_training_set[i]['bow']:
            vocab[w] = 1 if w not in vocab else vocab[w] + 1

    count_words_pairs = sorted([(v, [k for k in vocab if vocab[k] == v])
                                for v in set(vocab.values())],
                               key=itemgetter(0),
                               reverse=True)
    for pair in count_words_pairs:
        ordered_vocab.extend(sample(pair[1], len(pair[1])))
    return ordered_vocab


def build_feature_matrix(preprocessed_data_set, features, seed_weight=1.0):
    """Returns 2D array of size eqal to that of <preprocessed_data> \
        where each element is an array of size 2 + size of <features> and \
        format:
        [<feature_vector>, class_label, prediction_label(initialized to -1)]
    """
    return [[int(feature in preprocessed_data_set[i]['bow']) +
             int(feature == "BIAS")
             for feature in features] +
            [float(seed_weight)/float(len(preprocessed_data_set)),
             preprocessed_data_set[i]['class'], -1]
            for i in preprocessed_data_set]


def svm_train(x, weights_count):
    """Learn SVM from train matrix x
    """
    def class_label(b):
        return -1*int(not(b)) + b
    N = len(x)
    ETA_BY_N = ETA_SVM / float(N)
    w = [0.0] * weights_count
    is_misclassified = [False] * N
    for _ in xrange(DEFAULT_TRAINING_CYCLES):
        for i in xrange(len(x)):
            x[i][-1] = sum([w[j] * x[i][j] for j in range(weights_count)])
            is_misclassified[i] = bool(class_label(x[i][-2]) * x[i][-1] < 1)
        w_old = [wj for wj in w]
        for j in xrange(weights_count):
            w[j] = w[j] - ETA_BY_N * \
                          sum([LAMBDA * w[j] -
                               (class_label(x[i][-2]) * x[i][j]
                               if is_misclassified[i] else 0)
                               for i in range(N)])
        if 0.000001 > euclidean(w, w_old):
            break
    return w


def svm_classify(x, w):
    """Apply Learned SVM w to test matrix x
    """
    for i in xrange(len(x)):
        wx = sum([w[j] * x[i][j] for j in range(len(w))])
        x[i][-1] = int(wx / abs(wx))


def gini(m, x):
    """Compute Gini index for xth column of 2-D numpy array m over all rows
    """
    gini = 0.0
    s = len(m)
    absesnce_positives = 0
    absesnce_negatives = 0
    presence_positives = 0
    presence_negatives = 0
    for row in m:
        if row[x]:
            if row[-2]:
                presence_positives += row[-3]
            else:
                presence_negatives += row[-3]
        else:
            if row[-2]:
                absesnce_positives += row[-3]
            else:
                absesnce_negatives += row[-3]
    absesnce = absesnce_positives + absesnce_negatives
    presence = presence_positives + presence_negatives
    gini = \
        ((float((absesnce ** 2) -
                (absesnce_positives ** 2) - (absesnce_negatives ** 2)) /
          float(s * absesnce)) if absesnce else 0.0) + \
        ((float((presence ** 2) -
                (presence_positives ** 2) - (presence_negatives ** 2)) /
          float(s * presence)) if presence else 0.0)
    return gini


class TreeNode(object):
    """Tree Node for a STRICTLY BINARY tree
    """
    def __init__(self, class_label, lineage):
        """Node constructor
            @param class_label: class label which this leaf node predicts
            For non-leaf nodes:
                self.val holds the index of feature on which the node splits
            For leaf nodes:
                self.val holds the class label which it predicts
        """
        self.val = class_label
        self.lineage = lineage
        self.left = None
        self.right = None

    def is_leaf(self):
        """Returns: bool: True when node is leaf, False otherwise
        """
        return ((self.left is None) or (self.right is None))

    def expand(self, data, features, N,
               depth=DEFAULT_TREE_DEPTH, downsampling_power=1):
        """Adds 2 children by following splitting criteria
        """
        if not len(data):  # No data given
            self.val = -1
            return
        class_col = [row[-2] for row in data]
        class_label = int(class_col.count(1) > class_col.count(0))
        eligible_features = [f for f in features if f not in self.lineage]
        available_features = \
            sample(eligible_features,
                   int(round(len(eligible_features)**downsampling_power)))
        if len(self.lineage) <= depth \
           and (sum([row[-3] for row in data]) * N) > 9 \
           and len(available_features) > 0:  # eligible non-leaf node
            ginis = {i: gini(data, i) for i in available_features}
            split_idx = min(ginis, key=ginis.get)
            left_data = [row for row in data if row[split_idx] == 0]
            right_data = [row for row in data if row[split_idx] == 1]
            if sum([r[-3] for r in left_data]) and \
               sum([r[-3] for r in right_data]):  # Confirmed non-leaf node
                self.val = split_idx
                self.left = TreeNode(class_label, self.lineage + [split_idx])
                self.left.expand(left_data, features, N, depth=depth,
                                 downsampling_power=downsampling_power)
                self.right = TreeNode(class_label, self.lineage + [split_idx])
                self.right.expand(right_data, features, N, depth=depth,
                                  downsampling_power=downsampling_power)
            else:
                self.val = class_label
        else:
            self.val = class_label


def printTree(root):
    if root:
        print "|---" * len(root.lineage) + str(root.val) + \
            ("*" if root.is_leaf() else "")
        printTree(root.left)
        printTree(root.right)


def predict(vector, node):
    """predicts class
    """
    if not node:
        return -1
    if node.is_leaf():
        return node.val
    if vector[node.val]:
        return predict(vector, node.right)
    return predict(vector, node.left)


def dt_train(m, downsampling_power=1, tree_depth=DEFAULT_TREE_DEPTH):
    """trains a decision tree
    """
    features = range(len(m[0]) - 3)
    root = TreeNode(-1, [])
    root.expand(data=m, features=features, N=len(m), depth=tree_depth,
                downsampling_power=downsampling_power)
    return root


def dt_classify(m, root):
    """classifies the whole test matrix m
    """
    for i in xrange(len(m)):
        m[i][-1] = predict(m[i], root)


def bt_train(m, tree_depth=DEFAULT_TREE_DEPTH, tree_count=DEFAULT_TREE_COUNT):
    """trains t no. of models on pseudorandom samples from m
    """
    roots = []
    for _ in xrange(tree_count):
        roots.append(dt_train([choose_wr(m) for _ in xrange(len(m))],
                              tree_depth=tree_depth))
    return roots


def bt_classify(m, roots):
    """tests bagged models roots on m by assigning majority class
    """
    majority = len(roots) / 2
    for i in xrange(len(m)):
        m[i][-1] = int([predict(m[i], root)
                        for root in roots].count(1) > majority)


def rf_train(m, tree_depth=DEFAULT_TREE_DEPTH, tree_count=DEFAULT_TREE_COUNT):
    """trains t no. of models on pseudorandom samples with p**0.5 features \
        that are randomly selected from p features. The p**0.5 features may \
        overlap across t trees, but are unique within 1 tree)
    """
    roots = []
    range_m = range(len(m))
    for _ in xrange(tree_count):
        roots.append(dt_train([choose_wr(m) for _ in range_m],
                              downsampling_power=0.5,
                              tree_depth=tree_depth))
    return roots


def rf_classify(m, roots):
    """tests random forest models roots on m by assigning majority class
    """
    bt_classify(m, roots)


def ab_train(m,
             tree_depth=DEFAULT_TREE_DEPTH,
             tree_count=DEFAULT_TREE_COUNT):
    """trains t no. of models on pseudorandom samples with p**0.5 features \
        that are randomly selected from p features. The p**0.5 features may \
        overlap across t trees, but are unique within 1 tree)
    """
    range_m = range(len(m))
    models = []
    for i in range_m:
        m[i][-3] = 1.0 / float(len(m))
    for j in xrange(tree_count):
        root = dt_train(m)
        error_rate = 0.0
        temp_y = []
        for i in range_m:
            pred = predict(m[i], root)
            temp_y.append((1 if pred == 1 else -1))
            error_rate += ((1.0 if pred != m[i][-2] else 0.0) * m[i][-3])
        if error_rate == 0.0:
            break
        impact_factor = 0.5 * math.log((1 - error_rate)/error_rate, np.e)
        w_sum = 0.0
        for i in range_m:
            m[i][-3] = m[i][-3] * np.exp((-1) * impact_factor * temp_y[i] *
                                         (1 if m[i][-2] == 1 else -1))
            w_sum += m[i][-3]
        for i in range_m:
            m[i][-3] = float(m[i][-3]) / float(w_sum)
        models.append({'root': root, 'alpha': impact_factor})
    return models


def ab_classify(m, models):
    """tests random forest models roots on m by assigning majority class
    """
    range_m = range(len(m))
    for i in range_m:
        pred = [(model['alpha'] *
                (1 if predict(m[i], model['root']) == 1 else -1))
                for model in models]
        m[i][-1] = int(sum(pred) > 0.0)


def evaluate_zero_one_loss(test_matrix):
    """Returns zero-one loss from given result_matrix
    """
    return sum([int(example[-2] != int(example[-1] > 0))
                for example in test_matrix]) / float(len(test_matrix))


def run_model(**kwargs):
    """Controller function for step-wise execution of learning and application\
        of chosen model - LR OR SVM

        :Returns: (double) zero-one loss
        :kwargs:
            train_set:  training dataset dict
            test_set:   testing dataset dict
            model: {LR, SVM, NBC}
            feature_max_val: maximum value a feature can take (min value 0)
            console_print:  Flag: when True, prints zero-one loss
    """
    preprocessed_training_set = kwargs['train_set']
    preprocessed_testing_set = kwargs['test_set']
    feature_count = kwargs.get('feature_count', DEFAULT_FEATURE_COUNT)
    tree_depth = kwargs.get('tree_depth', DEFAULT_TREE_DEPTH)
    tree_count = kwargs.get('tree_count', DEFAULT_TREE_COUNT)
    ranked_vocabulary = build_ordered_vocabulary(preprocessed_training_set)
    features = sorted(ranked_vocabulary[DEFAULT_STOPWORD_COUNT:
                                        DEFAULT_STOPWORD_COUNT +
                                        feature_count])
    if kwargs['model'] == "SVM":
        features = ['BIAS'] + features
    train_matrix = build_feature_matrix(preprocessed_training_set, features)
    test_matrix = build_feature_matrix(preprocessed_testing_set, features)
    if kwargs['model'] == "SVM":
        svm_weights = svm_train(train_matrix, len(features))
        svm_classify(test_matrix, svm_weights)
    elif kwargs['model'] == "DT":
        root = dt_train(train_matrix, tree_depth=tree_depth)
        dt_classify(test_matrix, root)
    elif kwargs['model'] == "BT":
        roots = bt_train(train_matrix,
                         tree_depth=tree_depth,
                         tree_count=tree_count)
        bt_classify(test_matrix, roots)
    elif kwargs['model'] == "RF":
        roots = rf_train(train_matrix,
                         tree_depth=tree_depth,
                         tree_count=tree_count)
        rf_classify(test_matrix, roots)
    elif kwargs['model'] == "AB":
        models = ab_train(train_matrix,
                          tree_depth=tree_depth,
                          tree_count=tree_count)
        ab_classify(test_matrix, models)
    performance = evaluate_zero_one_loss(test_matrix)
    if kwargs.get('console_print', False):
        print "ZERO-ONE-LOSS-" + kwargs['model'], performance
    stdout.flush()
    return performance


def experiment(dataset, lookup, tss, feature_count,
               depth, tree_count, models):
    """
    """
    record = {}
    for M in models:
        performances = []
        for i in xrange(K):
            test_set = {k: dataset[k] for k in lookup[i]['test_set']}
            train_set = {k: dataset[k] for k in lookup[i]['train_set'][tss]}
            performance = run_model(train_set=train_set,
                                    test_set=test_set,
                                    feature_count=feature_count,
                                    tree_depth=depth,
                                    tree_count=tree_count,
                                    model=M,
                                    console_print=False)
            performances.append(performance)
#            print "\nTSS: ", tss, "\tFCOUNT: ", feature_count, "\tDEPTH: ", \
#                depth, "\tTCOUNT: ", tree_count, "\tMODEL: ", M, "\tRun: ", \
#                str(i + 1), "\tPERFORMANCE: ", performance
        record[M + '_mean'] = np.mean(performances)
        record[M + '_std_err'] = np.std(performances) / np.sqrt(K)
    return record


@timing
def analysis(filename, params=GIVEN_VALUES.keys()):
    """Run experiments with CV as specified in qurestion document
    """
    if not filename:
        return
    dataset = preprocess(read_tsv_file(filename))
    # Generate K disjoint folds with randomization
    random_seq = dataset.keys()
    shuffle(random_seq)
    evaluation_set = {}
    for i in xrange(K):
        s = i * len(dataset)/K
        evaluation_set[i] = {'test_set': random_seq[s:s + len(dataset)/K]}
        training_population = [idx for idx in random_seq
                               if idx not in evaluation_set[i]['test_set']]
        evaluation_set[i]['train_set'] = {}
        for tss in (GIVEN_VALUES['TSS'] if 'TSS' in params else [DEFAULT_TSS]):
            evaluation_set[i]['train_set'][tss] = \
                sample(training_population, int(tss * len(dataset)))
    for param in params:
        tss = DEFAULT_TSS
        feature_count = DEFAULT_FEATURE_COUNT
        depth = DEFAULT_TREE_DEPTH
        tree_count = DEFAULT_TREE_COUNT
        models = MODELS
        values = GIVEN_VALUES[param]
        sys.stdout.write("\n\nAnalysis by " + param + " " + str(values) + " ")
        record = {}
        for val in values:
            sys.stdout.write(" " + str(val))
            sys.stdout.flush()
            if param == 'TSS':
                tss = val
            if param == 'FCOUNT':
                feature_count = val
            if param == 'DEPTH':
                depth = val
                models = MODELS[:-1]
            if param == 'TCOUNT':
                tree_count = val
                models = MODELS[:-1]
            record[val] = experiment(dataset=dataset,
                                     lookup=evaluation_set,
                                     tss=tss,
                                     feature_count=feature_count,
                                     depth=depth,
                                     tree_count=tree_count,
                                     models=models)
        columns = [param] + [col for col in
                             [m + '_mean' for m in models] +
                             [m + '_std_err' for m in models]]
        rows = [[val] + [record[val][col] for col in
                [m + '_mean' for m in models] +
                [m + '_std_err' for m in models]] for val in record]
        rows.sort(key=itemgetter(0))
        if not path.isdir('output'):
            makedirs('output')
        with open(path.join('output', 'comparison_by_' +
                  param.lower() + '.csv'), "wb") as f:
            w = csv.writer(f)
            w.writerow(columns)
            print '\t\t'.join(columns)
            for row in rows:
                w.writerow(row)
                print '\t\t'.join([str(r) for r in row])
        draw_plot(columns, rows,
                  path.join('output', 'comparison'),
                  param, models)
        ttest(path.join('output', 'comparison_by_' +
                        param.lower() + '.csv'),
              path.join('output', 'ttests'),
              param, models)


def draw_plot(headers, record, output_path, param, models):
    """Plot learning curve with error bars
    """
    colors = ['red', 'blue', 'green', 'magenta', 'black']
    for i in range(len(models)):
        pyplot.errorbar(x=[row[0] for row in record],
                        y=[row[i + 1] for row in record],
                        yerr=[row[i + int(len(row)/2) + 1] for row in record],
                        color=colors[i],
                        label=models[i],
                        marker='o')
    pyplot.xlabel(param)
    pyplot.ylabel("ZERO-ONE-LOSS")
    pyplot.title('CS 573 Data Mining HW-4: Comparison of ' + ' '.join(models) +
                 ' on Text Classification\nBy: Parag Guruji,' +
                 ' pguruji@purdue.edu\nMean ZERO-ONE-LOSS vs ' + param +
                 ' with Std. Errors on error-bars\n',
                 loc='center')
    pyplot.xlim(pyplot.xlim()[0],
                pyplot.xlim()[1] + 0)
    pyplot.legend(loc='best', title='Legend: Models',
                  fancybox=True, framealpha=0.5)
    pyplot.savefig(output_path + '_by_' + param.lower() + '.png',
                   bbox_inches='tight')
    # pyplot.show()
    pyplot.gcf().clear()


def two_sample_ttest(m1, e1, n1, m2, e2, n2):
    """Returns P-value of 2-sample t-test on given summary data
        m(1/2): mean(1/2)
        e(1/2): standard error(1/2)
        n(1/2): sample size(1/2)
    """
    s1 = np.sqrt(n1) * e1
    s2 = np.sqrt(n2) * e2
    t2, p2 = ttest_ind_from_stats(m1, s1, n1,
                                  m2, s2, n2,
                                  equal_var=False)
    print "ttest stats: t = %g  p = %g" % (t2, p2)
    return t2, p2


def ttest(input_path, output_path, param, models, tail=1, bonferroni=True):
    """
    """
    with open(input_path, 'r') as datafile:
        datareader = csv.reader(datafile, delimiter=',')
        d = []
        out = [[param, "MODEL_1", "MODEL_2", "MEAN_1", "ERROR_1", "MEAN_2",
                "ERROR_2", "T", "P", "P < Alpha", "Result"]]
        for row in datareader:
            d.append(row)
        rows = [[float(i) for i in r] for r in d[1:]]
        for row in rows:
            for m1 in range(int(len(row)/2)):
                for m2 in range(int(len(row)/2))[m1 + 1:]:
                    t, p = two_sample_ttest(
                            row[m1 + 1], row[m1 + int(len(row)/2) + 1], K,
                            row[m2 + 1], row[m2 + int(len(row)/2) + 1], K)
                    str_in = "in"
                    str_dont = " DO NOT"
                    alpha = ALPHA_TTEST / float(len(rows) if bonferroni else 1)
                    if (p * tail / 2.0) < alpha:
                        str_in = ""
                    else:
                        str_dont = ""
                    result_str = "The difference in mean zero-one-loss of " + \
                        models[m1] + " and that of " + models[m2] + \
                        " is statistically " + str_in + "significant at " + \
                        str((1 - ALPHA_TTEST)*100) + "% Confidence Level." + \
                        " i.e. Both models materially" + str_dont + \
                        " perform the same."
                    out.append([row[0], models[m1], models[m2],
                                row[m1 + 1], row[m1 + int(len(row)/2) + 1],
                                row[m2 + 1], row[m2 + int(len(row)/2) + 1],
                                t, p, (p < ALPHA_TTEST),  result_str])
        with open(output_path + '_by_' + param.lower() + '.csv', "wb") as f:
            w = csv.writer(f)
            for row in out:
                w.writerow(row)
                print '\t\t'.join([str(r) for r in row])
        return out


if __name__ == "__main__":
    """Process commandline arguments and make calls to appropriate functions
    """
    parser = \
        argparse.ArgumentParser(
                    description='CS 573 Data Mining HW4 Ensembles',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('trainingDataFilename',
                        help='file-path of training set')
    parser.add_argument('testDataFilename',
                        help='file-path of testing dataset')
    parser.add_argument('modelIdx',
                        metavar='modelIdx',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        help='Choice of model: \
                        1. Decision Tree \
                        2. Bagging Tree  3. Random Forest  \
                        4. Support Vector Machine  \
                        5. Ada-Boosting on Decision Tree')
    parser.add_argument('-a', '--analysis',
                        metavar='analysis',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=None,
                        help="Choice of x variable of analysis: \
                        1. TSS = [0.025, 0.05, 0.125, 0.25] \
                        2. Feature Count = [200, 500, 1000, 1500] \
                        3. Tree Depth = [5, 10, 15, 20] \
                        4. Tree Count = [10, 25, 50, 100]  \
                        5. All 4")
    parser.add_argument('-t', '--ttest',
                        metavar='ttest',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=None,
                        help="Choice of x variable for ttests - \
                        attempts to read the summery stats from \
                        output/comparison_by_<lowercase param \
                        code>.csv: \
                        1. TSS (tss) = [0.025, 0.05, 0.125, 0.25] \
                        2. Feature Count (fcount) = [200, 500, 1000, 1500] \
                        3. Tree Depth (depth) = [5, 10, 15, 20] \
                        4. Tree Count (tcount) = [10, 25, 50, 100]  \
                        5. All 4")
    parser.add_argument('-p', '--plot',
                        metavar='plot',
                        type=int,
                        choices=[1, 2, 3, 4, 5],
                        default=None,
                        help="Choice of x variable for drawing plots - \
                        attempts to read the summery stats from \
                        output/comparison_by_<lowercase param \
                        code>.csv: \
                        1. TSS (tss) = [0.025, 0.05, 0.125, 0.25] \
                        2. Feature Count (fcount) = [200, 500, 1000, 1500] \
                        3. Tree Depth (depth) = [5, 10, 15, 20] \
                        4. Tree Count (tcount) = [10, 25, 50, 100]  \
                        5. All 4")
    args = parser.parse_args()
    if args.plot:
        if args.plot == 5:
            params = GIVEN_VALUES.keys()
        else:
            key = {1: 'TSS', 2: 'FCOUNT', 3: 'DEPTH', 4: 'TCOUNT'}
            params = [key[args.plot]]
        for param in params:
            models = MODELS if param in ['TSS', 'FCOUNT'] else MODELS[:-1]
            with open(path.join('output', 'comparison_by_' +
                      param.lower() + '.csv'), "r") as datafile:
                datareader = csv.reader(datafile, delimiter=',')
                lines = []
                for line in datareader:
                    lines.append(line)
                columns = lines[0]
                rows = [[float(r) for r in row] for row in lines[1:]]
                draw_plot(columns, rows,
                          path.join('output', 'comparison'),
                          param, models)
    elif args.ttest:
        if args.ttest == 5:
            params = GIVEN_VALUES.keys()
        else:
            key = {1: 'TSS', 2: 'FCOUNT', 3: 'DEPTH', 4: 'TCOUNT'}
            params = [key[args.ttest]]
        for param in params:
            models = MODELS if param in ['TSS', 'FCOUNT'] else MODELS[:-1]
            ttest(path.join('output',
                            'comparison_by_' + param.lower() + '.csv'),
                  path.join('output', 'ttests'),
                  param, models)
    elif args.analysis:
        if args.analysis == 5:
            params = GIVEN_VALUES.keys()
        else:
            key = {1: 'TSS', 2: 'FCOUNT', 3: 'DEPTH', 4: 'TCOUNT'}
            params = [key[args.analysis]]
        analysis(args.trainingDataFilename, params)
    else:
        training_set = preprocess(read_tsv_file(args.trainingDataFilename))
        testing_set = preprocess(read_tsv_file(args.testDataFilename))
        arguments = {'train_set': training_set,
                     'test_set': testing_set,
                     'model': MODELS[args.modelIdx - 1],
                     'console_print': True}
        run_model(**arguments)
