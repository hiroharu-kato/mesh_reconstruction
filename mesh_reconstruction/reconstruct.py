import argparse
import os
import random

import chainer
import cupy as cp
import neural_renderer
import numpy as np
import skimage.io

import models

RANDOM_SEED = 0
GPU = 0
DIRECTORY = './data/models'


def tile_images(images):
    rows = int(images.shape[0] ** 0.5)
    cols = int(images.shape[0] ** 0.5)
    image_size = images.shape[2]
    if images.ndim == 3:
        image = np.zeros((rows * image_size, cols * image_size))
    else:
        image = np.zeros((rows * image_size, cols * image_size, images.shape[1]))
        images = images.transpose((0, 2, 3, 1))
    for i in range(rows):
        for j in range(cols):
            image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = images[i * cols + j]
    return image


def run():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-eid', '--experiment_id', type=str)
    parser.add_argument('-d', '--directory', type=str, default=DIRECTORY)
    parser.add_argument('-i', '--input_image', type=str)
    parser.add_argument('-oi', '--output_image', type=str)
    parser.add_argument('-oo', '--output_obj', type=str)
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('-g', '--gpu', type=int, default=GPU)
    args = parser.parse_args()
    directory_output = os.path.join(args.directory, args.experiment_id)

    # set random seed, gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    cp.random.seed(args.seed)
    chainer.cuda.get_device(args.gpu).use()

    # load dataset
    image_in = skimage.io.imread(args.input_image).astype('float32') / 255
    if image_in.ndim != 3 or image_in.shape[-1] != 4:
        raise Exception('Input must be a RGBA image.')
    images_in = image_in.transpose((2, 0, 1))[None, :, :, :]
    images_in = chainer.cuda.to_gpu(images_in)

    # setup model & optimizer
    model = models.Model()
    model.to_gpu()
    chainer.serializers.load_npz(os.path.join(directory_output, 'model.npz'), model)

    # reconstruct .obj
    vertices, faces = model.reconstruct(images_in)
    neural_renderer.save_obj(args.output_obj, vertices.data.get()[0], faces.get()[0])

    # render reconstructed shape
    ones = chainer.cuda.to_gpu(np.ones((16,), 'float32'))
    distances = 2.732 * ones
    elevations = 30. * ones
    azimuths = chainer.cuda.to_gpu(np.arange(0, 360, 360. / 16.).astype('float32')) * ones
    viewpoints = neural_renderer.get_points_from_angles(distances, elevations, azimuths)
    images_out = model.reconstruct_and_render(chainer.functions.tile(images_in, (16, 1, 1, 1)), viewpoints)
    image_out = tile_images(images_out.data.get())
    image_out = (image_out * 255).clip(0, 255).astype('uint8')
    skimage.io.imsave(args.output_image, image_out)


if __name__ == '__main__':
    run()
