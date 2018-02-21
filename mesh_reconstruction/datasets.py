import os

import neural_renderer
import numpy as np
import tqdm


class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items()[0][1])
            voxels.append(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items()[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = np.zeros(batch_size, 'int32')
        viewpoint_ids_b = np.zeros(batch_size, 'int32')
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])
            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = self.images[data_ids_a].astype('float32') / 255.
        images_b = self.images[data_ids_b].astype('float32') / 255.

        distances = np.ones(batch_size, 'float32') * self.distance
        elevations = np.ones(batch_size, 'float32') * self.elevation
        viewpoints_a = neural_renderer.get_points_from_angles(distances, elevations, -viewpoint_ids_a * 15)
        viewpoints_b = neural_renderer.get_points_from_angles(distances, elevations, -viewpoint_ids_b * 15)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids
        for i in range((data_ids.size - 1) / batch_size + 1):
            images = self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.
            voxels = self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] / 24]
            yield images, voxels
