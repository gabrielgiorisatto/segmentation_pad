import os
import functools
import tensorflow as tf
import tensorflow.contrib as tfcontrib
import rgb_lab_formulation as Conv_img
import multiprocessing
from sklearn.model_selection import train_test_split
import sys
import numpy as np

sys.path.append('utils/')
from augmentation import aug as jd


img_shape = (256, 256, 7)
batch_size = 8
epochs = 5

def _process_pathnames(fname, label_path, cspace, img_shape):
    # We map this function onto each pathname pair
    img_str = tf.read_file(fname)
    # Load Image in range [0...255]
    img = tf.image.decode_jpeg(img_str, channels=3)
    # Convert to range [0...1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize Image
    if(tf.shape(img)[0] != 256 or tf.shape(img)[1] != 256):
        img = tf.image.resize_images(img, [img_shape[0], img_shape[1]])

    label_img_str = tf.read_file(label_path)
    # Load image in range [0...255]
    label_img = tf.image.decode_png(label_img_str, channels=3)
    # Load image in range [0...1]
    label_img = tf.image.convert_image_dtype(label_img, tf.float32)
    # The label image should only have values of 1 or 0, indicating pixel wise
    # object (car) or not (background). We take the second channel only.
    label_img = label_img[:, :, 1]
    label_img = tf.expand_dims(label_img, axis=-1)
    # Resize Image
    if(tf.shape(label_img)[0] != 256 or tf.shape(label_img)[1] != 256):
        label_img = tf.image.resize_images(label_img, [img_shape[0], img_shape[1]])
    return img, label_img


# ## Shifting the image

def shift_img(output_img, label_img, width_shift_range, height_shift_range):
    """This fn will perform the horizontal or vertical shift"""
    if width_shift_range or height_shift_range:
        if width_shift_range:
            width_shift_range = tf.random_uniform(
                [], -width_shift_range * img_shape[1],
                width_shift_range * img_shape[1])
        if height_shift_range:
            height_shift_range = tf.random_uniform(
                [], -height_shift_range * img_shape[0],
                height_shift_range * img_shape[0])
        # Translate both
        output_img = tfcontrib.image.translate(
            output_img, [width_shift_range, height_shift_range])
        label_img = tfcontrib.image.translate(
            label_img, [width_shift_range, height_shift_range])
    return output_img, label_img


# ## Flipping the image randomly 

def flip_img(horizontal_flip, tr_img, label_img):
    if horizontal_flip:
        flip_prob = tf.random_uniform([], 0.0, 1.0)
        tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                    lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                    lambda: (tr_img, label_img))
    return tr_img, label_img


# ## Assembling our transformations into our augment function

def _augment(img,
        label_img,
        cspace='RGB',
        augment=True):  # Randomly translate the image vertically
    seed = np.random.randint(2**32 - 1, dtype=np.int64)
    # print('Seed = {}'.format(seed))
    if(augment):
        img, label_img = jd.transform_img(img, label_img, flip_left_right=True, flip_up_down=True, 
                   crop=0.75, rot90=True, brightness=None, blur=False,
                   contrast=None, hue=None, gamma=None,
                   saturation=None, noise=None, size=(256,256,3))
      #img = tf.image.random_hue(img, 0.1)
      #img, label_img = flip_img(True, img, label_img)
      #img, label_img = shift_img(img, label_img, 0.1,
      #                           0.1)
    if (cspace == 'RGB'):
        img = img
    elif (cspace == 'HSV'):
        img = tf.image.rgb_to_hsv(img)
    elif (cspace == 'RGB-HSV'):
        img = tf.concat([img, tf.image.rgb_to_hsv(img)], -1)
    elif (cspace == 'LAB'):
        img_lab = Conv_img.rgb_to_lab(img)
        L, a, b = Conv_img.preprocess_lab(img_lab)
        img = tf.stack([L,a,b], axis=2)
    elif (cspace == 'RGB-LAB'):
        img_lab = Conv_img.rgb_to_lab(img)
        L, a, b = Conv_img.preprocess_lab(img_lab)
        img_lab = tf.stack([L,a,b], axis=2)
        img = tf.concat([img, img_lab], -1)
    elif (cspace == 'RGB-HSV-LAB'):
        img_rgb_hsv = tf.concat([img, tf.image.rgb_to_hsv(img)], -1)
        img_lab = Conv_img.rgb_to_lab(img)
        L, a, b = Conv_img.preprocess_lab(img_lab)
        img_lab = tf.stack([L,a,b], axis=2)
        img = tf.concat([img_rgb_hsv, img_lab], -1)
    elif (cspace == 'RGB-HSV-L'):
        img_rgb_hsv = tf.concat([img, tf.image.rgb_to_hsv(img)], -1)
        img_lab = Conv_img.rgb_to_lab(img)
        L, a, b = Conv_img.preprocess_lab(img_lab)
        img = tf.concat([img_rgb_hsv, tf.stack([L], axis=2)], -1)
    elif (cspace == 'RGB-SV-LAB'):
        img_hsv = tf.image.rgb_to_hsv(img)
        H, S, V = tf.unstack(img_hsv, axis=2)
        img_sv = tf.stack([S,V], axis=2)
        img_rgb_sv = tf.concat([img, img_sv], -1)
        img_lab = Conv_img.rgb_to_lab(img)
        L, a, b = Conv_img.preprocess_lab(img_lab)
        img_lab = tf.stack([L,a,b], axis=2)
        img = tf.concat([img_rgb_sv, img_lab], -1)
    return img, label_img

def get_baseline_dataset(filenames,
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5,
                         batch_size=batch_size,
                         shuffle=True,
                         cspace='RGB',
                         img_shape=(256,256,3)):
    num_x = len(filenames)
    # Create a dataset from the filenames and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # Map our preprocessing function to every element in our dataset, taking
    # advantage of multithreading
    dataset = dataset.map(lambda x, y: _process_pathnames(x, y, cspace, img_shape), num_parallel_calls=multiprocessing.cpu_count())

    dataset = dataset.map(preproc_fn, num_parallel_calls=threads)

    if shuffle:
        dataset = dataset.shuffle(num_x)

    # It's necessary to repeat our data for all epochs
    dataset = dataset.repeat().batch(batch_size)
    return dataset

def prepare_train_val(dataset='ISIC', cspace='RGB', img_shape=(256, 256, 3), batch_size=8):
    ## Data Import
    if (dataset == 'ISIC'):
        data_path = 'data/dataset/raw/'
        img_dir = os.path.join(data_path, 'train')
        images = os.listdir(img_dir)

        x_train_filenames = []
        y_train_filenames = []

        for image_name in images:
            if 'mask' in image_name:
                continue
            x_train_filenames.append(img_dir + '/' + image_name)
            y_train_filenames.append(img_dir + '/' + 'mask' +
                                     image_name.split('.')[0] + '.png')
    elif (dataset == 'ISIC_TEST'):
        data_path = 'dataset/ISIC/'
        train_dir = os.path.join(data_path, 'data')
        mask_dir = os.path.join(data_path, 'mask')
        images = os.listdir(train_dir)
        x_train_filenames = []
        for image_name in sorted(images):
            x_train_filenames.append(train_dir + '/' + image_name)
        images = os.listdir(mask_dir)
        y_train_filenames = []
        for image_name in sorted(images):
            y_train_filenames.append(mask_dir + '/' + image_name)
    elif (dataset == 'PAD'):
        data_path = 'dataset/pad_separado/train/'
        train_dir = os.path.join(data_path, 'data')
        mask_dir = os.path.join(data_path, 'mask')
        x_train_filenames = []
        for folder in os.listdir(train_dir):
            path = os.path.join(train_dir, folder)
            for image_name in sorted(os.listdir(path)):
                x_train_filenames.append(path + '/' + image_name)
        y_train_filenames = []
        for folder in os.listdir(mask_dir):
            path = os.path.join(mask_dir, folder)
            for image_name in sorted(os.listdir(path)):
                y_train_filenames.append(path + '/' + image_name)

    x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = \
                        train_test_split(x_train_filenames, y_train_filenames, test_size=0.2)

    num_train_examples = len(x_train_filenames)
    num_val_examples = len(x_val_filenames)

    print("Number of training examples: {}".format(num_train_examples))
    print("Number of validation examples: {}".format(num_val_examples))

    tr_cfg = {
    'cspace': cspace
    }

    tr_preprocessing_fn = functools.partial(_augment, **tr_cfg)

    train_ds = get_baseline_dataset(
        x_train_filenames,
        y_train_filenames,
        preproc_fn=tr_preprocessing_fn,
        batch_size=batch_size,
        cspace=cspace,
        img_shape=img_shape)

    val_cfg = {
    'augment': False,
    'cspace': cspace
    }

    val_preprocessing_fn = functools.partial(_augment, **val_cfg)

    val_ds = get_baseline_dataset(
        x_val_filenames,
        y_val_filenames,
        preproc_fn=val_preprocessing_fn,
        batch_size=batch_size,
        cspace=cspace,
        img_shape=img_shape)
    return train_ds, val_ds, num_train_examples, num_val_examples

def prepare_test(dataset='PAD', cspace='RGB', img_shape=(256, 256, 3)):
    if (dataset == 'PAD'):
        data_path = 'dataset/pad_separado/test/'
        test_dir = os.path.join(data_path, 'data')
        mask_dir = os.path.join(data_path, 'mask')
        x_test_filenames = []
        for folder in os.listdir(test_dir):
            path = os.path.join(test_dir, folder)
            for image_name in sorted(os.listdir(path)):
                x_test_filenames.append(path + '/' + image_name)
        y_test_filenames = []
        for folder in os.listdir(mask_dir):
            path = os.path.join(mask_dir, folder)
            for image_name in sorted(os.listdir(path)):
                y_test_filenames.append(path + '/' + image_name)
    elif (dataset == 'ISIC_TEST'):
        data_path = '/pg/data/ISIC2017_REDUZIDO/'
        test_dir = os.path.join(data_path, 'data')
        mask_dir = os.path.join(data_path, 'mask')
        images = os.listdir(test_dir)
        x_test_filenames = []
        for image_name in sorted(images):
            x_test_filenames.append(test_dir + '/' + image_name)
        images = os.listdir(mask_dir)
        y_test_filenames = []
        for image_name in sorted(images):
            y_test_filenames.append(mask_dir + '/' + image_name.split('.')[0] + '.png')
    elif (dataset == 'ISIC'):
        data_path = 'dataset/'
        test_dir = os.path.join(data_path, 'raw/test/ISIC')
        mask_dir = os.path.join(data_path, 'expected/ISIC')
        images = os.listdir(test_dir)
        x_test_filenames = []
        for image_name in sorted(images):
            x_test_filenames.append(test_dir + '/' + image_name)
        images = os.listdir(mask_dir)
        y_test_filenames = []
        for image_name in sorted(images):
            y_test_filenames.append(mask_dir + '/' + image_name.split('.')[0] + '.png')

    num_test_examples = len(x_test_filenames)
    print("Number of testing examples: {}".format(num_test_examples))

    test_cfg = {
    'augment': False,
    'cspace': cspace
    }

    test_preprocessing_fn = functools.partial(_augment, **test_cfg)

    test_ds = get_baseline_dataset(
        x_test_filenames,
        y_test_filenames,
        preproc_fn=test_preprocessing_fn,
        batch_size=1,
        shuffle=False,
        cspace=cspace,
        img_shape=img_shape)
    return test_ds, num_test_examples
