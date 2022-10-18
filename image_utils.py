import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import display as display_fn
from imageio import mimsave
from IPython.display import Image, clear_output


def tensor_to_image(tensor):
    '''converts a tensor to an image'''
    tensor_shape = tf.shape(tensor)
    number_elem_shape = tf.shape(tensor_shape)
    if number_elem_shape > 3:
        assert tensor_shape[0] == 1
        tensor = tensor[0]
    return tf.keras.preprocessing.image.array_to_img(tensor)


def load_img(path_to_img):
    '''loads an image as a tensor and scales it to 512 pixels'''
    max_dim = 512
    image = tf.io.read_file(path_to_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, (max_dim, max_dim))
    image = image[tf.newaxis, :]

    image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


def load_images(content_path, style_path):
    '''loads the content and path images as tensors'''
    content_image = load_img("{}".format(content_path))
    style_image = load_img("{}".format(style_path))

    return content_image, style_image


def imshow(image, title=None):
    '''displays an image with a corresponding title'''
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def show_images_with_objects(images, titles=[]):
    '''displays a row of images with corresponding titles'''
    if len(images) != len(titles):
        return

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)
    plt.savefig('sample.jpg', bbox_inches='tight')


def display_gif(gif_path):
    '''displays the generated images as an animated gif'''
    with open(gif_path,'rb') as f:
        display_fn(Image(data=f.read(), format='png'))


def create_gif(gif_path, images):
    '''creates animation of generated images'''
    mimsave(gif_path, images, fps=1)

    return gif_path


def clip_image_values(image, min_value=0.0, max_value=255.0):
    '''clips the image pixel values by the given min and max'''
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


def preprocess_image(image):
    '''centers the pixel values of a given image to use with VGG-19'''
    image = tf.cast(image, dtype=tf.float32)
    image = tf.keras.applications.vgg19.preprocess_input(image)

    return image


# Plot Utilities

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var


def plot_deltas_for_single_image(x_deltas, y_deltas, name="Original", row=1):
    plt.figure(figsize=(14,10))
    plt.subplot(row,2,1)
    plt.yticks([])
    plt.xticks([])

    clipped_y_deltas = clip_image_values(2*y_deltas+0.5, min_value=0.0, max_value=1.0)
    imshow(clipped_y_deltas, "Horizontal Deltas: {}".format(name))

    plt.subplot(row,2,2)
    plt.yticks([])
    plt.xticks([])

    clipped_x_deltas = clip_image_values(2*x_deltas+0.5, min_value=0.0, max_value=1.0)
    imshow(clipped_x_deltas, "Vertical Deltas: {}".format(name))


def plot_deltas(original_image_deltas, stylized_image_deltas):
    orig_x_deltas, orig_y_deltas = original_image_deltas

    stylized_x_deltas, stylized_y_deltas = stylized_image_deltas

    plot_deltas_for_single_image(orig_x_deltas, orig_y_deltas, name="Original")
    plot_deltas_for_single_image(stylized_x_deltas, stylized_y_deltas, name="Stylized Image", row=2)
