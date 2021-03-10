"""Splits the training set into a training and validation set for use in the first stage, during which
we do not have access to the true validation set labels."""
import argparse
import csv
import random
from collections import Counter, defaultdict

import numpy as np
from sklearn.preprocessing import LabelEncoder


# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def main(args):
    with open(args.train_labels_file) as orig_file:
        signers = []
        samples = []
        labels = []
        reader = csv.reader(orig_file)
        for row in reader:
            signers.append(row[0].split('_')[0])
            samples.append(row[0])
            labels.append(row[1])

        labels_enc = LabelEncoder().fit_transform(labels)

        cv = stratified_group_k_fold(samples, labels_enc, signers, 5, seed=42)
        train_indices, val_indices = next(cv)

        new_samples = []
        for i in range(len(samples)):
            if i in train_indices:
                new_samples.append((samples[i], labels[i], 'train'))
            else:
                new_samples.append((samples[i], labels[i], 'val'))

        with open(args.train_labels_file.replace('train_labels.csv', 'train_val_labels_STAGE1.csv'), 'w') as new_file:
            writer = csv.writer(new_file)
            writer.writerows(new_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_labels_file', type=str)
    args = parser.parse_args()
    main(args)
