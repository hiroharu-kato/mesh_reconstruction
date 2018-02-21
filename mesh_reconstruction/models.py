import chainer
import chainer.functions as cf
import chainer.links as cl
import neural_renderer

import loss_functions
import renderer
import voxelization


class Encoder(chainer.Chain):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        with self.init_scope():
            dim_hidden = [dim1 * 2 ** 0, dim1 * 2 ** 1, dim1 * 2 ** 2, dim2, dim2]
            self.conv1 = cl.Convolution2D(dim_in, dim_hidden[0], 5, stride=2, pad=2)
            self.conv2 = cl.Convolution2D(dim_hidden[0], dim_hidden[1], 5, stride=2, pad=2)
            self.conv3 = cl.Convolution2D(dim_hidden[1], dim_hidden[2], 5, stride=2, pad=2)
            self.linear1 = cl.Linear(dim_hidden[2] * 8 * 8, dim_hidden[3])
            self.linear2 = cl.Linear(dim_hidden[3], dim_hidden[4])
            self.linear3 = cl.Linear(dim_hidden[4], dim_out)

    def __call__(self, x):
        x = cf.relu(self.conv1(x))
        x = cf.relu(self.conv2(x))
        x = cf.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))
        x = cf.relu(self.linear1(x))
        x = cf.relu(self.linear2(x))
        x = cf.relu(self.linear3(x))
        return x


class Decoder(chainer.Chain):
    def __init__(
            self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0,
            centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()

        with self.init_scope():
            self.vertices_base, self.faces = neural_renderer.load_obj(filename_obj)
            self.num_vertices = self.vertices_base.shape[0]
            self.num_faces = self.faces.shape[0]
            self.centroid_scale = centroid_scale
            self.bias_scale = bias_scale
            self.obj_scale = 0.5

            dim = 1024
            dim_hidden = [dim, dim * 2]
            self.linear1 = cl.Linear(dim_in, dim_hidden[0])
            self.linear2 = cl.Linear(dim_hidden[0], dim_hidden[1])
            self.linear_centroids = cl.Linear(dim_hidden[1], 3)
            self.linear_bias = cl.Linear(dim_hidden[1], self.num_vertices * 3)
            self.linear_centroids.W.lr = centroid_lr
            self.linear_centroids.b.lr = centroid_lr
            self.linear_bias.W.lr = bias_lr
            self.linear_bias.b.lr = bias_lr

    def to_gpu(self):
        super(Decoder, self).to_gpu()
        self.vertices_base = chainer.cuda.to_gpu(self.vertices_base)
        self.faces = chainer.cuda.to_gpu(self.faces)

    def __call__(self, x):
        h = cf.relu(self.linear1(x))
        h = cf.relu(self.linear2(h))
        centroids = self.linear_centroids(h) * self.centroid_scale
        bias = self.linear_bias(h) * self.bias_scale
        bias = cf.reshape(bias, (-1, self.num_vertices, 3))

        base = self.vertices_base * self.obj_scale
        base = self.xp.broadcast_to(base[None, :, :], bias.shape)

        sign = self.xp.sign(base)
        base = self.xp.absolute(base)
        base = self.xp.log(base / (1 - base))

        centroids = cf.broadcast_to(centroids[:, None, :], bias.shape)
        centroids = cf.tanh(centroids)
        scale_pos = 1 - centroids
        scale_neg = centroids + 1

        vertices = cf.sigmoid(base + bias)
        vertices = vertices * sign
        vertices = cf.relu(vertices) * scale_pos - cf.relu(-vertices) * scale_neg
        vertices += centroids
        vertices *= 0.5
        faces = self.xp.tile(self.faces[None, :, :], (x.shape[0], 1, 1))

        return vertices, faces


class Model(chainer.Chain):
    def __init__(self, filename_obj='./data/obj/sphere_642.obj', lambda_smoothness=0.):
        super(Model, self).__init__()
        self.lambda_smoothness = lambda_smoothness
        self.vertices_predicted_a = None
        self.vertices_predicted_b = None
        with self.init_scope():
            self.encoder = Encoder()
            self.decoder = Decoder(filename_obj)
            self.smoothness_loss_parameters = loss_functions.smoothness_loss_parameters(self.decoder.faces)

            self.renderer = renderer.Renderer()
            self.renderer.image_size = 64
            self.renderer.viewing_angle = 15.
            self.renderer.anti_aliasing = True

    def to_gpu(self, device=None):
        super(Model, self).to_gpu()
        self.smoothness_loss_parameters = [chainer.cuda.to_gpu(p) for p in self.smoothness_loss_parameters]

    def predict(self, images_a, images_b, viewpoints_a, viewpoints_b):
        batch_size = images_a.shape[0]
        images = self.xp.concatenate((images_a, images_b), axis=0)
        viewpoints = self.xp.concatenate((viewpoints_a, viewpoints_a, viewpoints_b, viewpoints_b), axis=0)
        self.renderer.eye = viewpoints

        vertices, faces = self.decoder(self.encoder(images))  # [a, b]
        vertices_c = cf.concat((vertices, vertices), axis=0)  # [a, b, a, b]
        faces_c = cf.concat((faces, faces), axis=0).data  # [a, b, a, b]
        silhouettes = self.renderer.render_silhouettes(vertices_c, faces_c)  # [a/a, b/a, a/b, b/b]
        silhouettes_a_a = silhouettes[0 * batch_size:1 * batch_size]
        silhouettes_b_a = silhouettes[1 * batch_size:2 * batch_size]
        silhouettes_a_b = silhouettes[2 * batch_size:3 * batch_size]
        silhouettes_b_b = silhouettes[3 * batch_size:4 * batch_size]
        vertices_a = vertices[:batch_size]
        vertices_b = vertices[batch_size:]
        return silhouettes_a_a, silhouettes_b_a, silhouettes_a_b, silhouettes_b_b, vertices_a, vertices_b

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces

    def reconstruct_and_render(self, images_in, viewpoints):
        self.renderer.eye = viewpoints
        vertices, faces = self.reconstruct(images_in)
        textures = self.xp.ones((viewpoints.shape[0], faces.shape[1], 2, 2, 2, 3), 'float32')
        images_out = self.renderer.render(vertices, faces, textures)

        return images_out

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.decoder(self.encoder(images))
        faces = neural_renderer.vertices_to_faces(vertices, faces).data
        faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
        voxels_predicted = voxelization.voxelize(faces, 32, False)
        voxels_predicted = voxels_predicted.transpose((0, 2, 1, 3))[:, :, :, ::-1]
        iou = (voxels * voxels_predicted).sum((1, 2, 3)) / (0 < (voxels + voxels_predicted)).sum((1, 2, 3))
        return iou

    def __call__(self, images_a, images_b, viewpoints_a, viewpoints_b):
        # predict vertices and silhouettes
        silhouettes_a_a, silhouettes_b_a, silhouettes_a_b, silhouettes_b_b, vertices_a, vertices_b = (
            self.predict(images_a, images_b, viewpoints_a, viewpoints_b))

        # compute loss
        loss_silhouettes = (
                               loss_functions.iou_loss(images_a[:, 3, :, :], silhouettes_a_a) +
                               loss_functions.iou_loss(images_a[:, 3, :, :], silhouettes_b_a) +
                               loss_functions.iou_loss(images_b[:, 3, :, :], silhouettes_a_b) +
                               loss_functions.iou_loss(images_b[:, 3, :, :], silhouettes_b_b)) / 4
        if self.lambda_smoothness != 0:
            loss_smoothness = (
                                  loss_functions.smoothness_loss(vertices_a, self.smoothness_loss_parameters) +
                                  loss_functions.smoothness_loss(vertices_b, self.smoothness_loss_parameters)) / 2
        else:
            loss_smoothness = 0
        loss = loss_silhouettes + self.lambda_smoothness * loss_smoothness

        # report
        loss_list = {
            'loss_silhouettes': loss_silhouettes,
            'loss_smoothness': loss_smoothness,
            'loss': loss,
        }
        chainer.reporter.report(loss_list, self)

        return loss
