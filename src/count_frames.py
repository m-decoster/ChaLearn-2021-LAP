"""For every video file, saves an additional file signerX_sampleY_nframes
which contains a single integer (in text) with the number of frames."""
import argparse
import glob
import os

import torchvision


def main(args):
    for dataset in ['train', 'val', 'test']:
        videos = glob.glob(os.path.join(args.input_dir, dataset, '*_color.mp4'))
        for video_file in videos:
            frames, _, _ = torchvision.io.read_video(video_file, pts_unit='sec')
            with open(video_file.replace('color.mp4', 'nframes'), 'w') as of:
                of.write(f'{frames.size(0)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    args = parser.parse_args()
    main(args)
