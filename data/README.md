# Data directory

This directory contains original and derived label files.

## Original files

- `train_labels.csv`: original train labels
- `val_labels.csv`: original validation labels
- `predictions_test_template.csv`: prediction template file for test set
- `predictions_validation_template.csv`: prediction template file for validation set

## Derived files

- `train_val_labels_STAGE1.csv`: train and validation labels for the development stage, derived from `train_labels.csv` using `stage1.py`. This is a stratified grouped split (80-20%).
- `train_val_labels_STAGE2.csv`: train and validation labels for the final stae, derived from `train_labels.csv` and `val_labels.csv` using `stage2.py`. This is a concatenated file with added `train` and `val` labels indicating the source subset.

## Usage

To reproduce our development stage results, use the `train_val_labels_STAGE1.csv` file. To reproduce our test stage results, use the `train_val_labels_STAGE2.csv` file.

**Note** that in the development stage, both the training and "validation" samples
originate from the training set, and therefore the code in the datasets
must be modified to read "validation" samples from the training folder.
You can do this by modifying the `job_path` argument in the dataset constructor
for the validation dataset.

For example, change [these few lines](https://github.com/m-decoster/ChaLearn-2021-LAP/blob/prep/src/datasets/handcrop_poseflow.py#L56) to say

```python3
self.val_set = ChaLearnDataset(self.data_dir, 'train', 'val',
                               os.path.join(self.data_dir, '..', '..', 'train_val_labels_STAGE1.csv'),
                               transform,
                               self.sequence_length, self.temporal_stride)
```
