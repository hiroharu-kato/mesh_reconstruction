import chainer
import cupy as cp


def _voxelize_sub1(faces, size, dim=2):
    assert (0 <= dim)
    bs, nf = faces.shape[:2]
    if dim == 0:
        i = cp.array([2, 1, 0])
        faces = faces[:, :, :, i]
    elif dim == 1:
        i = cp.array([0, 2, 1])
        faces = faces[:, :, :, i]
    faces = cp.ascontiguousarray(faces)
    voxels = cp.zeros((faces.shape[0], size, size, size), 'int32')
    chainer.cuda.elementwise(
        'int32 j, raw T faces, raw int32 bs, raw int32 nf, raw int32 vs',
        'raw int32 voxels',
        '''
            int y = j % vs;
            int x = (j / vs) % vs;
            int bn = j / (vs * vs);

            //
            for (int fn = 0; fn < nf; fn++){
                float* face = &faces[(bn * nf + fn) * 9];
                float y1d = face[3] - face[0];
                float x1d = face[4] - face[1];
                float z1d = face[5] - face[2];
                float y2d = face[6] - face[0];
                float x2d = face[7] - face[1];
                float z2d = face[8] - face[2];
                float ypd = y - face[0];
                float xpd = x - face[1];
                float det = x1d * y2d - x2d * y1d;
                if (det == 0) continue;
                float t1 = (y2d * xpd - x2d * ypd) / det;
                float t2 = (-y1d * xpd + x1d * ypd) / det;
                if (t1 < 0) continue;
                if (t2 < 0) continue;
                if (1 < t1 + t2) continue;
                int zi = floor(t1 * z1d + t2 * z2d + face[2]);

                int yi, xi;
                yi = y;
                xi = x;
                if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
                    voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
                yi = y - 1;
                xi = x;
                if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
                    voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
                yi = y;
                xi = x - 1;
                if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
                    voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
                yi = y - 1;
                xi = x - 1;
                if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
                    voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
            }
        ''',
        'function',
    )(cp.arange(bs * size * size).astype('int32'), faces, bs, nf, size, voxels)
    voxels = voxels.swapaxes(dim + 1, -1)
    return voxels


def _voxelize_sub2(faces, size):
    bs, nf = faces.shape[:2]
    faces = cp.ascontiguousarray(faces)
    voxels = cp.zeros((faces.shape[0], size, size, size), 'int32')
    chainer.cuda.elementwise(
        'int32 j, raw T faces, raw int32 bs, raw int32 nf, raw int32 vs',
        'raw int32 voxels',
        '''
            int fn = j % nf;
            int bn = j / nf;
            float* face = &faces[(bn * nf + fn) * 9];
            for (int k = 0; k < 3; k++) {
                int yi = face[3 * k + 0];
                int xi = face[3 * k + 1];
                int zi = face[3 * k + 2];
                if ((0 <= yi) && (yi < vs) && (0 <= xi) && (xi < vs) && (0 <= zi) && (zi < vs))
                    voxels[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] = 1;
            }
        ''',
        'function',
    )(cp.arange(bs * nf).astype('int32'), faces, bs, nf, size, voxels)
    return voxels


def _voxelize_sub3(voxels):
    # fill in
    bs, vs = voxels.shape[:2]
    voxels = cp.ascontiguousarray(voxels)
    visible = cp.zeros_like(voxels, 'int32')
    chainer.cuda.elementwise(
        'int32 j, raw int32 bs, raw int32 vs',
        'raw int32 voxels, raw int32 visible',
        '''
            int z = j % vs;
            int x = (j / vs) % vs;
            int y = (j / (vs * vs)) % vs;
            int bn = j / (vs * vs * vs);
            int pn = j;
            if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) {
                if (voxels[pn] == 0) visible[pn] = 1;
            }
        ''',
        'function',
    )(cp.arange(bs * vs * vs * vs).astype('int32'), bs, vs, voxels, visible)

    sum_visible = visible.sum()
    while True:
        chainer.cuda.elementwise(
            'int32 j, raw int32 bs, raw int32 vs',
            'raw int32 voxels, raw int32 visible',
            '''
                int z = j % vs;
                int x = (j / vs) % vs;
                int y = (j / (vs * vs)) % vs;
                int bn = j / (vs * vs * vs);
                int pn = j;
                if ((y == 0) || (y == vs - 1) || (x == 0) || (x == vs - 1) || (z == 0) || (z == vs - 1)) return;
                if (voxels[pn] == 0 && visible[pn] == 0) {
                    int yi, xi, zi;
                    yi = y - 1;
                    xi = x;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y + 1;
                    xi = x;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x - 1;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x + 1;
                    zi = z;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x;
                    zi = z - 1;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                    yi = y;
                    xi = x;
                    zi = z + 1;
                    if (visible[bn * vs * vs * vs + yi * vs * vs + xi * vs + zi] != 0) visible[pn] = 1;
                }
            ''',
            'function',
        )(cp.arange(bs * vs * vs * vs).astype('int32'), bs, vs, voxels, visible)
        if visible.sum() == sum_visible:
            break
        else:
            sum_visible = visible.sum()
    return 1 - visible


def voxelize(faces, size, normalize=False):
    faces = cp.copy(faces)
    if normalize:
        faces -= faces.min((0, 1, 2), keepdims=True)
        faces /= faces.max()
        faces *= 1. * (size - 1) / size
        margin = 1 - faces.max((0, 1, 2))
        faces += margin[None, None, None, :] / 2
        faces *= size
    else:
        faces *= size

    voxels0 = _voxelize_sub1(faces, size, 0)
    voxels1 = _voxelize_sub1(faces, size, 1)
    voxels2 = _voxelize_sub1(faces, size, 2)
    voxels3 = _voxelize_sub2(faces, size)
    voxels = voxels0 + voxels1 + voxels2 + voxels3
    voxels = (0 < voxels).astype('int32')
    voxels = _voxelize_sub3(voxels)

    return voxels
