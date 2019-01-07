import numpy as np
import matplotlib as mpl
mpl.use('agg')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12, 12)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.python.keras import models, backend as K
import time
from unet import unet, bce_dice_loss, dice_loss, dice_coeff
from data import prepare_train_val, prepare_test
import rgb_lab_formulation as Conv_img
import utils
import gc
import os
# Train your model

# cspace = "RGB" or "HSV" or "HSV-RGB" or "LAB"
def callback_def(epochs, cspace, type_train='', it=0, lr=0.001, batch_size=8):
    if(not os.path.exists('models/' + type_train + '/' + cspace + '/')):
        os.makedirs('models/' + type_train + '/' + cspace + '/')
    save_model_path = 'models/' + type_train + '/' + cspace + '/weights' + str(epochs) + '_' + str(it) + '_' + str(lr) + '_' + str(batch_size) + '.hdf5'
    save_log_path = 'models/' + type_train + '/' + cspace + '/log' + str(epochs) + '_' + str(it) + '_' + str(lr) + '_' + str(batch_size) + '_' + str(int(time.time()))
    cp = [tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model_path,
        monitor='val_dice_loss',
        save_best_only=True,
        verbose=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=save_log_path
        )]
    return cp, save_model_path

# cspace = "RGB" or "HSV" or "HSV-RGB" or "LAB"
def train_history(epochs, cspace, history, type_train='', it=0):
    dice = history.history['dice_loss']
    val_dice = history.history['val_dice_loss']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    fig = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, dice, label='Training Dice Loss')
    plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Dice Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    fig.savefig('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/loss.png', bbox_inches='tight')

# x_in and y_in are numpy arrays
def fjaccard(x_in, y_in):
    x = x_in.flatten()
    y = y_in.flatten()
    return np.sum(np.logical_and(x, y).astype(float)) / np.sum(
        np.logical_or(x, y).astype(float))

def evaluate_test(model, test_ds, num_test_examples, cspace, epochs, save_model_path=None, type_train='',write_images=True, it=0):
    if (save_model_path != None):
        model = models.load_model(
            save_model_path,
            custom_objects={
                'bce_dice_loss': bce_dice_loss,
                'dice_loss': dice_loss
            })
    # Let's visualize some of the outputs
    mjccard = 0
    score = 0
    v_jaccard = np.zeros(num_test_examples)
    v_sensitivity = np.zeros(num_test_examples)
    v_specificity = np.zeros(num_test_examples)
    v_accuracy = np.zeros(num_test_examples)
    v_dice = np.zeros(num_test_examples)

    crf_jaccard = np.zeros(num_test_examples)
    crf_sensitivity = np.zeros(num_test_examples)
    crf_specificity = np.zeros(num_test_examples)
    crf_accuracy = np.zeros(num_test_examples)
    crf_dice = np.zeros(num_test_examples)

    data_aug_iter = test_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()
    # if(write_images):
    if(not os.path.exists('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/predict/')):
            os.makedirs('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/predict/')
    for j in range(num_test_examples):
        # Running next element in our graph will produce a batch of images
        batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
        img = batch_of_imgs[0]

        predicted_label = model.predict(batch_of_imgs)[0]
        mpimg.imsave('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/predict/' + str(j) + '.png', predicted_label[:,:,0])
        mask_pred = (predicted_label[:, :, 0] > 0.55).astype(int)
        label = label.astype(int)

        v_jaccard[j] = fjaccard(label[0, :, :, 0], mask_pred)
        v_sensitivity[j] = utils.sensitivity(label[0,:,:,0], mask_pred)
        v_specificity[j] = utils.specificity(label[0,:,:,0], mask_pred)
        v_accuracy[j] = utils.accuracy(label[0,:,:,0], mask_pred)
        v_dice[j] = utils.dice_coeff(label[0,:,:,0], mask_pred)
        score += v_jaccard[j] if v_jaccard[j] >= 0.65 else 0
        print(score)
        mjccard += v_jaccard[j]

        img_rgb = img[:, :, :3]

        # if(cspace == 'HSV'):
        #     img_rgb = tf.keras.backend.get_session().run(tf.image.hsv_to_rgb(img_rgb))
        # elif(cspace == 'LAB'):
        #     img_rgb = tf.keras.backend.get_session().run(Conv_img.lab_to_rgb(img_rgb))

        crf_mask = utils.dense_crf(np.array(img_rgb*255).astype(np.uint8), np.array(predicted_label[:, :, 0]).astype(np.float32))

        crf_jaccard[j] = fjaccard(label[0, :, :, 0], crf_mask)
        crf_sensitivity[j] = utils.sensitivity(label[0,:,:,0], crf_mask)
        crf_specificity[j] = utils.specificity(label[0,:,:,0], crf_mask)
        crf_accuracy[j] = utils.accuracy(label[0,:,:,0], crf_mask)
        crf_dice[j] = utils.dice_coeff(label[0,:,:,0], crf_mask)

        if(write_images):
            fig = plt.figure(figsize=(25, 25))

            plt.subplot(1, 4, 1)
            plt.imshow(img[:, :, :3])
            plt.title("Input image")
            
            plt.subplot(1, 4, 2)
            plt.imshow(label[0, :, :, 0])
            plt.title("Actual Mask")
            
            plt.subplot(1, 4, 3)
            plt.imshow(predicted_label[:, :, 0] > 0.55)
            plt.title("Predicted Mask\n" +
                        "Jaccard = " + str(v_jaccard[j]) +
                        '\nSensitivity = ' + str(v_sensitivity[j]) +
                        '\nSpecificity = ' + str(v_specificity[j]) +
                        '\nAccuracy = ' + str(v_accuracy[j]) +
                        '\nDice = ' + str(v_dice[j]))
            
            plt.subplot(1, 4, 4)
            plt.imshow(crf_mask)
            plt.title("CRF Mask\n" +
                        "Jaccard = " + str(crf_jaccard[j]) +
                        '\nSensitivity = ' + str(crf_sensitivity[j]) +
                        '\nSpecificity = ' + str(crf_specificity[j]) +
                        '\nAccuracy = ' + str(crf_accuracy[j]) +
                        '\nDice = ' + str(crf_dice[j]))
            
            fig.savefig(
                'results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/' + str(j) + '.png',
                bbox_inches='tight')
            plt.close(fig)
            mpimg.imsave('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/predict/' + str(j) + '.png', predicted_label[:,:,0])
            plt.close()

    mjccard /= num_test_examples
    score /= num_test_examples
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/jaccard', v_jaccard)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/sensitivity', v_sensitivity)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/specificity', v_specificity)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/accuracy', v_accuracy)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/dice', v_dice)
    with open('results/' + type_train + cspace + '/' + str(epochs)  + '/' + str(it) + '/score','w') as f:
        f.write('Score = ' + str(score) +
        '\nSensitivity = ' + str(np.mean(v_sensitivity)) +
        '\nSpecificity = ' + str(np.mean(v_specificity)) +
        '\nAccuracy = ' + str(np.mean(v_accuracy)) +
        '\nDice = ' + str(np.mean(v_dice)) +
        '\nJaccars = ' + str(np.mean(v_jaccard)))

    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/crf_jaccard', crf_jaccard)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/crf_sensitivity', crf_sensitivity)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/crf_specificity', crf_specificity)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/crf_accuracy', crf_accuracy)
    np.savetxt('results/' + type_train + cspace + '/' + str(epochs) + '/' + str(it) + '/crf_dice', crf_dice)
    with open('results/' + type_train + cspace + '/' + str(epochs)  + '/' + str(it) + '/crf_score','w') as f:
        f.write('Sensitivity = ' + str(np.mean(crf_sensitivity)) +
        '\nSpecificity = ' + str(np.mean(crf_specificity)) +
        '\nAccuracy = ' + str(np.mean(crf_accuracy)) +
        '\nDice = ' + str(np.mean(crf_dice)) +
        '\nJaccars = ' + str(np.mean(crf_jaccard)))

    print('Jccard = ' + str(mjccard))
    print('Score = ' + str(score))
    return mjccard, score

def testa_aug(test_ds):
    data_aug_iter = test_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()
    for i in range(16):
        batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
        img = batch_of_imgs[0]
        plt.subplot(4,4,i+1)
        plt.imshow(img[:, :, :3])
        # plt.title("Input image")
        plt.hold(True)
        plt.contour(label[0, :, :, 0])
        # plt.title("Actual Mask")
        plt.hold(False)
    plt.show()

if __name__ == '__main__':
    epochs = 5
    cspacev = ["RGB"]
    img_shapev = [3]
    img_shape = (256, 256, 3)
    type_train = 'direct_train'
    batch_size = 8
    if type_train == 'fine_tune':
        lr = 0.00001
    else:
        lr = 0.0001
    i = 0
    for cspace, img_dim in zip(cspacev, img_shapev):
        inicio = 0
        fim = 1
        for it in range(inicio,fim):
            with tf.Graph().as_default():
                with tf.Session().as_default():
                    # Adjust image shape
                    imagelst = list(img_shape)
                    imagelst[2] = img_dim
                    img_shape = tuple(imagelst)
                    
                    print("Loading dataset...\n")
                    
                    if(type_train == "direct_transfer"):
                        dataset = 'ISIC'
                    else:
                        dataset = 'PAD'
                    train_ds, val_ds, num_train_examples, num_val_examples = prepare_train_val(
                            dataset=dataset,
                            cspace=cspace,
                            img_shape=img_shape,
                            batch_size=1)
                    pad_test_ds, pad_num_test_examples = prepare_test(
                            dataset='PAD',
                            cspace=cspace,
                            img_shape=img_shape)
                    
                    print("Done. Iteration = " + str(it) + "\n")
                    # Load Model
                    i = i + 1
                    if(type_train == "fine_tune" or type_train == 'ISIC/fine_tune'):
                        model = models.load_model(
                        'models/' + 'direct_transfer' + '/' + cspace + '/weights' + str(epochs) + '_' + str(it) + '_' + '0.0001' + '_' + str(batch_size) + '.hdf5',
                        custom_objects={
                            'bce_dice_loss': bce_dice_loss,
                            'dice_loss': dice_loss
                        })
                    else:
                        model = unet(img_shape, lr=lr)
                    start_time = time.time()
                    cp, save_model_path = callback_def(
                            epochs, cspace,
                            type_train=type_train, it=it, lr=lr, batch_size=batch_size)
                    
                    
                    # Train Model
                    history = model.fit(
                            train_ds,
                            steps_per_epoch=int(np.ceil(num_train_examples / float(batch_size))),
                            epochs=epochs,
                            validation_data=val_ds,
                            validation_steps=int(np.ceil(num_val_examples / float(batch_size))),
                            callbacks=cp)
                    
                    # Save Results
                    if(not os.path.exists('results/' + type_train + '/' + cspace + '/' + str(epochs) + '/' + str(it))):
                        os.makedirs('results/' + type_train + '/' + cspace + '/' + str(epochs) + '/' + str(it))
                    with open('results/' + type_train + '/' + cspace + '/' + str(epochs) + '/' + str(it) + '/time','w') as f:
                        f.write(str(time.time() - start_time))
                    train_history(
                            epochs,
                            cspace,
                            history,
                            type_train= type_train + '/',
                            it=it)
                    
                   # Test Model
                    evaluate_test(
                            model,
                            pad_test_ds,
                            pad_num_test_examples,
                            cspace,
                            epochs,
                            type_train= type_train + '/',
                            write_images=True,
                            it=it,
                            save_model_path='models/' + type_train + '/' + cspace + '/weights' + str(epochs) + '_' + str(it) + '_' + str(lr) + '_' + str(batch_size) + '.hdf5',)
                    # evaluate_test(model, isictest_test_ds, isictest_num_test_examples, cspace, epochs,
                    #     save_model_path='models/' + cspace + '/weights' + str(epochs) + '.hdf5',
                    #     ISIC='/ISICTEST')
                    # evaluate_test(model, isic_test_ds, isic_num_test_examples, cspace, epochs,
                    #     save_model_path='models/' + cspace + '/weights' + str(epochs) + '.hdf5',
                    #     ISIC='/ISIC')
            K.clear_session()
            del model, pad_test_ds
            del model, train_ds, val_ds,pad_test_ds
            gc.collect()