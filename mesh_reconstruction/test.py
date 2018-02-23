import argparse
import os
import random

import chainer
import cupy as cp
import numpy as np

import datasets
import models
import training

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASS_NAMES = {
    '02691156': 'airplane',
    '02828884': 'bench',
    '02933112': 'dresser',
    '02958343': 'car',
    '03001627': 'chair',
    '03211117': 'display',
    '03636649': 'lamp',
    '03691459': 'speaker',
    '04090263': 'rifle',
    '04256520': 'sofa',
    '04379243': 'table',
    '04401088': 'phone',
    '04530566': 'vessel',
}
RANDOM_SEED = 0
GPU = 0
DIRECTORY = './data/models'
MODEL_DIRECTORY = './data/models'
DATASET_DIRECTORY = './data/dataset'


def run():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--experiment_id', type=str)
    parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-cls', '--class_ids', type=str, default=CLASS_IDS_ALL)
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('-g', '--gpu', type=int, default=GPU)
    args = parser.parse_args()
    directory_output = os.path.join(args.model_directory, args.experiment_id)

    # set random seed, gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    cp.random.seed(args.seed)
    chainer.cuda.get_device(args.gpu).use()

    # load dataset
    dataset_test = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'test')

    # setup model & optimizer
    model = models.Model()
    model.to_gpu()
    chainer.serializers.load_npz(os.path.join(directory_output, 'model.npz'), model)

    # evaluate
    reporter = chainer.Reporter()
    observation = {}
    with reporter.scope(observation):
        training.validation(None, model, dataset_test)
    for key in sorted(observation.keys()):
        key_display = key
        for class_id in CLASS_NAMES.keys():
            key_display = key_display.replace(class_id, CLASS_NAMES[class_id])
        print '%s: %.4f' % (key_display, observation[key])


if __name__ == '__main__':
    run()
