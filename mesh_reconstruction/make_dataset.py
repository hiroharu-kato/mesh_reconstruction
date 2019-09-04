#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import dataset
import scipy.misc

image_size = 64
batch_size = 64
distance = 2.732
input_mask = True

class_ids = [
    '03001627', '02691156', '02828884', '02933112', '02958343', '03211117', '03636649', '03691459', '04090263',
    '04256520', '04379243', '04401088', '04530566']
sets = ['train', 'val', 'test']
path = '/home/mil/kato/large_data/projection/reconstruction'
# path = '/media/disk2/lab/projection/reconstruction'

for image_postfix in ['blender', 'box', 'bilinear']:
    for mask_postfix in ['blender', 'box', 'bilinear']:
        directory_out = os.path.join(path, 'dataset_%s_%s' % (image_postfix, mask_postfix))
        if os.path.exists(directory_out):
            continue
        os.makedirs(directory_out)
        for class_id in class_ids:
            for set_name in sets:
                filename_ids = os.path.join('../../resource', 'shapenetcore_ids/%s_%sids.txt' % (class_id, set_name))
                ids = open(filename_ids).readlines()
                ids = [i.split('/')[-1].strip() for i in ids if len(i) != 0]
                # ids = ids[:10]

                num_data = len(ids)
                images = np.zeros((num_data, 24, image_size, image_size, 3), 'uint8')
                masks = np.zeros((num_data, 24, image_size, image_size), 'uint8')
                voxels = np.zeros((num_data, 32, 32, 32), 'bool')

                for i, object_id in enumerate(ids):
                    print 'loading data (%s %s %d / %d)' % (class_id, set_name, i, len(ids))

                    # load voxel
                    filename_voxel = os.path.join(path, 'shapenet_voxels_32/%s/%s.npz' % (class_id, object_id,))
                    voxels[i] = np.load(filename_voxel).items()[0][1]

                    for k, azimuth in enumerate(range(0, 360, 15)):
                        filename_image = os.path.join(path, 'shapenet_images_64_%.1f_%s/%s/%s/e030_a%03d.png' % (
                            distance, image_postfix, class_id, object_id, azimuth))
                        image = scipy.misc.imread(filename_image)[:, :, :3]
                        mask = scipy.misc.imread(filename_image)[:, :, 3]
                        mask = mask.astype('float32') / mask.max()
                        image = image.astype('float32') / 255.
                        image = image * mask[:, :, None] + np.ones_like(image) * (1 - mask[:, :, None])
                        image = 1 - image  # !!
                        image = (image * 255).astype('uint8')
                        images[i, k] = image

                        filename_image = os.path.join(path, 'shapenet_images_64_%.1f_%s/%s/%s/e030_a%03d.png' % (
                            distance, mask_postfix, class_id, object_id, azimuth))
                        mask = scipy.misc.imread(filename_image)[:, :, 3]
                        mask = mask.astype('float32') / mask.max()
                        masks[i, k] = (mask * 255).astype('uint8')

                images = images.transpose((0, 1, 4, 2, 3))
                if input_mask:
                    images = np.concatenate((images, masks[:, :, None, :, :]), axis=2)

                np.savez_compressed(os.path.join(directory_out, '%s_%s_images.npz' % (class_id, set_name)), images)
                np.savez_compressed(os.path.join(directory_out, '%s_%s_masks.npz' % (class_id, set_name)), masks)
                np.savez_compressed(os.path.join(directory_out, '%s_%s_voxels.npz' % (class_id, set_name)), voxels)