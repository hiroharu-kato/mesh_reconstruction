import argparse
import functools
import os
import random

import chainer
import cupy as cp
import neural_renderer
import numpy as np

import datasets
import models
import training

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LR_REDUCE_POINT = 0.95
NUM_ITERATIONS = 1000000
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
LAMBDA_SMOOTHNESS = 0

LOG_INTERVAL = 10000
RANDOM_SEED = 0
GPU = 0
MODEL_DIRECTORY = './data/models'
DATASET_DIRECTORY = './data/dataset'


def run():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--experiment_id', type=str)
    parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
    parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)
    parser.add_argument('-cls', '--class_ids', type=str, default=CLASS_IDS_ALL)
    parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-ls', '--lambda_smoothness', type=float, default=LAMBDA_SMOOTHNESS)
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('-lrp', '--lr_reduce_point', type=float, default=LR_REDUCE_POINT)
    parser.add_argument('-ni', '--num_iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('-li', '--log_interval', type=int, default=LOG_INTERVAL)
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
    dataset_train = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'train')
    dataset_val = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'val')
    train_iter = training.MyIterator(dataset_train, args.batch_size)

    # setup model & optimizer
    model = models.Model(lambda_smoothness=args.lambda_smoothness)
    model.to_gpu()
    optimizer = neural_renderer.Adam(args.learning_rate)
    optimizer.setup(model)

    # setup trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer, converter=training.my_convertor)
    trainer = chainer.training.Trainer(updater, stop_trigger=(args.num_iterations, 'iteration'), out=directory_output)
    trainer.extend(chainer.training.extensions.LogReport(trigger=(args.log_interval, 'iteration')))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['iteration', 'main/loss_silhouettes', 'main/loss_smoothness', 'val/iou', 'elapsed_time']))
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))
    trainer.extend(
        functools.partial(training.validation, model=model, dataset=dataset_val),
        name='validation',
        priority=chainer.training.PRIORITY_WRITER,
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(
        functools.partial(
            training.lr_shift, optimizer=optimizer, iterations=[args.num_iterations * args.lr_reduce_point]),
        name='lr_shift',
        trigger=(1, 'iteration'))

    # main loop
    trainer.run()

    # save model
    chainer.serializers.save_npz(os.path.join(directory_output, 'model.npz'), model)


if __name__ == '__main__':
    run()
