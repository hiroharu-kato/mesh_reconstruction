import chainer.functions as cf

import neural_renderer

DEFAULT_IMAGE_SIZE = 256
DEFAULT_ANTI_ALIASING = True
DEFAULT_NEAR = 0.1
DEFAULT_FAR = 100
DEFAULT_EPS = 1e-3
DEFAULT_BACKGROUND_COLOR = (0, 0, 0)
USE_UNSAFE_IMPLEMENTATION = False


def rasterize_silhouettes(
        faces,
        image_size=DEFAULT_IMAGE_SIZE,
        anti_aliasing=DEFAULT_ANTI_ALIASING,
        near=DEFAULT_NEAR,
        far=DEFAULT_FAR,
        eps=DEFAULT_EPS,
        background_color=DEFAULT_BACKGROUND_COLOR,
):
    if anti_aliasing:
        # 2x super-sampling
        faces = faces * (2 * image_size - 1) / (2 * image_size - 2)
        images = neural_renderer.Rasterize(
            image_size * 2, near, far, eps, background_color, return_rgb=False, return_alpha=True, return_depth=False)(
            faces)[1]
    else:
        images = neural_renderer.Rasterize(
            image_size, near, far, eps, background_color, return_rgb=False, return_alpha=True, return_depth=False)(
            faces)[1]

    # transpose & vertical flip
    images = images[:, ::-1, :]

    if anti_aliasing:
        # 0.5x down-sampling
        images = cf.resize_images(images[:, None, :, :], (image_size, image_size))[:, 0]

    return images


class Renderer(neural_renderer.Renderer):
    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = neural_renderer.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = neural_renderer.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = neural_renderer.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        images = rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images
