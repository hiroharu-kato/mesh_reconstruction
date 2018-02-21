import chainer
import numpy as np


def my_convertor(data, device=None):
    # data is tuple of numpy array
    return tuple([chainer.cuda.to_gpu(d.astype('float32'), device) for d in data])


class MyIterator(chainer.dataset.Iterator):
    # iterator for training

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self.epoch = 0
        self.epoch_detail = 0

    def __next__(self):
        return self.dataset.get_random_batch(self.batch_size)

    next = __next__


def validation(trainer=None, model=None, dataset=None):
    # evaluate voxel IoUs on all classes
    with chainer.configuration.using_config('train', False):
        ious = {}
        for class_id in dataset.class_ids:
            iou = 0
            for batch in dataset.get_all_batches_for_evaluation(100, class_id):
                batch = my_convertor(batch)
                iou += model.evaluate_iou(*batch).sum()
            iou /= dataset.num_data[class_id] * 24
            ious['%s/iou_%s' % (dataset.set_name, class_id)] = iou
        ious['%s/iou' % dataset.set_name] = np.mean([float(v) for v in ious.values()])
        chainer.report(ious)


def lr_shift(trainer=None, optimizer=None, iterations=None, factor=0.1):
    if trainer.updater.iteration in iterations:
        optimizer.alpha *= factor
