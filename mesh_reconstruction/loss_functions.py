import chainer.functions as cf
import numpy as np


def iou(data1, data2):
    # target, prediction
    axes = tuple(range(data1.ndim)[1:])
    intersection = cf.sum(data1 * data2, axis=axes)
    union = cf.sum(data1 + data2 - data1 * data2, axis=axes)
    return cf.sum(intersection / union) / intersection.size


def iou_loss(data1, data2):
    return 1 - iou(data1, data2)


def smoothness_loss_parameters(faces):
    if hasattr(faces, 'get'):
        faces = faces.get()
    vertices = list(set([tuple(v) for v in np.sort(np.concatenate((faces[:, 0:2], faces[:, 1:3]), axis=0))]))

    v0s = np.array([v[0] for v in vertices], 'int32')
    v1s = np.array([v[1] for v in vertices], 'int32')
    v2s = []
    v3s = []
    for v0, v1 in zip(v0s, v1s):
        count = 0
        for face in faces:
            if v0 in face and v1 in face:
                v = np.copy(face)
                v = v[v != v0]
                v = v[v != v1]
                if count == 0:
                    v2s.append(int(v[0]))
                    count += 1
                else:
                    v3s.append(int(v[0]))
    v2s = np.array(v2s, 'int32')
    v3s = np.array(v3s, 'int32')
    return v0s, v1s, v2s, v3s


def smoothness_loss(vertices, parameters, eps=1e-6):
    # make v0s, v1s, v2s, v3s
    v0s, v1s, v2s, v3s = parameters
    batch_size = vertices.shape[0]

    v0s = vertices[:, v0s, :]
    v1s = vertices[:, v1s, :]
    v2s = vertices[:, v2s, :]
    v3s = vertices[:, v3s, :]

    a1 = v1s - v0s
    b1 = v2s - v0s
    a1l2 = cf.sum(cf.square(a1), axis=-1)
    b1l2 = cf.sum(cf.square(b1), axis=-1)
    a1l1 = cf.sqrt(a1l2 + eps)
    b1l1 = cf.sqrt(b1l2 + eps)
    ab1 = cf.sum(a1 * b1, axis=-1)
    cos1 = ab1 / (a1l1 * b1l1 + eps)
    sin1 = cf.sqrt(1 - cf.square(cos1) + eps)
    c1 = a1 * cf.broadcast_to((ab1 / (a1l2 + eps))[:, :, None], a1.shape)
    cb1 = b1 - c1
    cb1l1 = b1l1 * sin1

    a2 = v1s - v0s
    b2 = v3s - v0s
    a2l2 = cf.sum(cf.square(a2), axis=-1)
    b2l2 = cf.sum(cf.square(b2), axis=-1)
    a2l1 = cf.sqrt(a2l2 + eps)
    b2l1 = cf.sqrt(b2l2 + eps)
    ab2 = cf.sum(a2 * b2, axis=-1)
    cos2 = ab2 / (a2l1 * b2l1 + eps)
    sin2 = cf.sqrt(1 - cf.square(cos2) + eps)
    c2 = a2 * cf.broadcast_to((ab2 / (a2l2 + eps))[:, :, None], a2.shape)
    cb2 = b2 - c2
    cb2l1 = b2l1 * sin2

    cos = cf.sum(cb1 * cb2, axis=-1) / (cb1l1 * cb2l1 + eps)

    loss = cf.sum(cf.square(cos + 1)) / batch_size
    return loss
