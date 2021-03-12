# ChaLearn 2021 LAP

This repository contains the code of my submission to the ChaLearn 2021 Looking At People: Signer Independent Isolated Sign Language Recognition challenge.

| Track | Development Score | Test Score |
|-------|-------------------|------------|
| RGB   | 0.9083            | 0.9292     |
| RGB-D | 0.9167            | 0.9332     |

This PyTorch code allows you to reproduce my results by training the model
yourself, or by using the provided pre-trained model weights in the [Releases section](https://github.com/m-decoster/ChaLearn-2021-LAP/releases).

If you wish to train the models yourself, I recommend that you download the OpenPose keypoints and pose flow files from the Releases section, and use those
along with the original MP4 files. This saves you the trouble of extracting them yourself using OpenPose.

The process to reproducing my results is explained below.

1. [Set up virtual environment with requirements](#requirements)
2. [Prepare the dataset folder](#dataset-preparations)
3. [Download and extract the keypoints and pose flow](#using-prepared-files)
    1. Alternatively, [extract the keypoints yourself using OpenPose](#reproducing-results-from-scratch)
4. [Train the model](#training-the-models)
5. [Perform inference to obtain the predictions](#inference)

If you use the pre-trained model checkpoints, you can skip step 4.

## Requirements

This code base has following dependencies:

- Python 3.8.5
- PyTorch 1.7.1
- Torchvision 0.8.2 with PyAV 8.0.2
- PyTorch Lightning 1.1.1
- OpenCV-Python 4.3.0.36

Older and newer versions of these dependencies may work as well but are not tested.

I recommend you create a virtual environment and install the dependencies using:

```bash
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip   # Make sure you have the latest version of pip
pip install -r src/requirements.txt
```

### Hardware

For GPUs with small memory capacity, you can use the reduce the batch size
by a factor `n` and use the `--accumulate-grad-batches n` option during training.
We use `--batch_size 4 --accumulate-grad-batches 8` to emulate a batch size of 32.

*Using more than 1 GPU is currently **not** supported! If you run out of VRAM, use the above option.*

## Dataset preparations

If you wish to reproduce our results, you will need to prepare some folders for the data.
First, create the following directory hierarchy (the absolute location of `project` does not matter and neither does it name):

```
project/
project/data
project/data/mp4
project/data/kp
project/data/kpflow2
```

This can be done using the command

```bash
mkdir -p project/data/{mp4,kp,kpflow2}
```

Copy the CSV files from the `data/` directory in this repository to the `project/` directory.
These are the prediction templates and the label files.
See also `data/README.md` for more information.

Create a `train`, `val` and `test` directory under `project/data/mp4`.

```bash
mkdir -p project/data/mp4/{train,val,test}
```

Place the corresponding MP4 files there. They can be found on the [competition website](http://chalearnlap.cvc.uab.es/dataset/40/description/#).

These directories will also be created under `kp` and `kpflow2` when the corresponding
feature extraction code is executed. You do not need to manually create them.

Finally, run the `count_frames.py` script with as argument the path to the `mp4` directory, e.g.,

```bash
python count_frames.py -i project/data/mp4
```

Your dataset is now prepared to either extract the keypoint files yourself, or use the ones provided by us.

## Using prepared files

We provide keypoint and pose flow files in the [Releases section](https://github.com/m-decoster/ChaLearn-2021-LAP/releases) to allow for easy reproduction of the results.
Note that these are large archives, so I have split them in sections. You can extract them using

```bash
cat kp.tar.bz2.* | tar -jxv
cat kpflow2.tar.bz2.* | tar -jxv
```

Then you can proceed with the dataset preparations below without needing to run OpenPose and pose flow extraction yourself.
That is, you can skip the next section and go directly to [training the models](#training-the-models).

## Reproducing results from scratch

You can also reproduce our results from scratch, starting from the MP4 files.
In this case, you will need to extract keypoints and pose flow yourself.

### Keypoint extraction

You can extract OpenPose using the OpenPose demo and following command (modify `'0,1'` to match the available GPUs on your machine).
First you need to install the [OpenPose demo](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md) and download the [BODY-135 model](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/experimental_models#single-network-whole-body-pose-estimation-model).

```bash
import glob
import os

all_files = glob.glob('project/data/mp4/*/*_color.mp4')

CALL_STRING = 'CUDA_VISIBLE_DEVICES={} ./openpose.bin --render_pose 0 --number_people_max 1 --display 0 --video {} --write_json {} --model_pose BODY_135'

for sample in all_files:
    out_dir = sample.replace('mp4', 'kp').replace('_color.mp4', '')
    os.makedirs(out_dir, exist_ok=True)
    c = CALL_STRING.format('0,1', sample, out_dir)
    os.system(c)
```

Keypoints will be available as directories of JSON files in a `kp` directory on the same level as the `mp4` directory.

The model also requires pose flow, which can be computed from these keypoint files.

```bash
cd src
python extract_poseflow.py --input_dir project/data/kp
```

Pose flow will be available as `.npy` files in a `kpflow2` directory on the same level as the `mp4` and `kp` directories.

## Training the models

For training, you should create a log directory, to which the experiment details
as well as Tensorboard event files will be written. We will assume that this
log directory exists at `$LOG_DIR`.

In our case we use 4 workers for the dataset loading, but you can set this according
to your CPU's capacity. We will assume that this is set as `$NUM_WORKERS`.

You should download the data yourself and provide the path to our scripts, specifically
the path to the `.mp4` files. We will assume that this data directory exists at `$DATA_DIR`.
So for the explanation above,

```console
$ echo $DATA_DIR
project/data/mp4
```

## Training (RGB)

For training on RGB data, use this command.

```bash
python -m train --log_dir $LOG_DIR --model VTN_HCPF --dataset handcrop_poseflow --num_workers $NUM_WORKERS \
    --data_dir $DATA_DIR --sequence_length 16 --temporal_stride 2 --learning_rate 1e-4 \
    --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 8 \
    --batch_size 4 --accumulate-grad-batches 8
```

## Training (RGB-D)

For training on RGB-D data, use this command.

```bash
python -m train --log_dir $LOG_DIR --model VTN_HCPF_D --dataset handcrop_poseflow_rgbd --num_workers $NUM_WORKERS \
    --data_dir $DATA_DIR --sequence_length 16 --temporal_stride 2 --learning_rate 1e-4 \
    --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 8 \
    --batch_size 4 --accumulate-grad-batches 8
```

## Inference

The `predict.py` script can be used. This requires an additional prediction template file provided by the challenge organizers. We will assume this file exists at `$PREDICTION_TEMPLATE`.

After training, you should have a checkpoint file at `$CHECKPOINT_PATH`. You can predict using

```bash
python -m predict --log_dir $LOG_DIR --model $MODEL --dataset $DATASET --num_workers $NUM_WORKERS \
    --data_dir $DATA_DIR --sequence_length 16 --temporal_stride 2 --learning_rate 1e-4 \
    --gpus 1 --cnn rn34 --num_layers 4 --num_heads 8 --max_epochs $NUM_EPOCHS \
    --checkpoint=$CHECKPOINT_PATH --submission_template $PREDICTION_TEMPLATE --out predictions.csv \
    --batch_size 4
```

for the `$MODEL` and `$DATASET` of your choice.
Alternatively, you can use one of the pre-trained models provided in the [Releases section](https://github.com/m-decoster/ChaLearn-2021-LAP/releases)
(the command remains the same).

## Prediction files

This repository also provides prediction files for the models on both the validation and the test set. These can be found under the `predictions` directory.

## LICENCE

This code is available under the MIT licence (see LICENCE). Part of the code base is based on the Intel OpenVINO toolkit (see LICENCE\_OPENVINO).
