"""
train function step 1
"""

import argparse
import os
import keras
import keras.preprocessing.image
import keras.backend as K
from model import combinedmodel
from callback.common import RedirectModel
from preprocess.combined_data import CombinedDataGenerator
import tensorflow as tf
from model import accum_optimizer


def create_generators(args):
    """ Create generators for training and validation. """
    train_generator = CombinedDataGenerator(
        data_dir=args.dataset_path,
        set_name='train',
        batch_size=args.batch_size,
        sample_num=args.sample_num
    )

    validation_generator = CombinedDataGenerator(
        data_dir=args.dataset_path,
        set_name='val',
        batch_size=args.batch_size,
        sample_num=args.sample_num
    )
    return train_generator, validation_generator


def create_models(sample_num, class_num, lr=1e-5):
    """ Creates three models (model, training_model)."""

    # load anchor parameters, or pass None (so that defaults will be used)
    model = combinedmodel.CombinedModel(sample_num=sample_num, class_num=class_num)
    training_model = model
    # compile model
    training_model.compile(
        loss='binary_crossentropy',
        optimizer=accum_optimizer.AccumOptimizer(keras.optimizers.adam(lr=lr, clipnorm=0.001), 100),
        metrics=['accuracy']
    )
    return model, training_model


def create_callbacks(model, args):
    """ Creates the callbacks to use during training."""
    callbacks = []

    # callback 1
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=args.tensorboard_dir,
        histogram_freq=0,
        batch_size=args.batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None,
    )
    callbacks.append(tensorboard_callback)

    # callback 2  save the model
    if not args.is_snapshot:
        # ensure directory created first; otherwise h5py will error after epoch.
        try:
            os.makedirs(args.snapshot_path)
        except OSError:
            if not os.path.isdir(args.snapshot_path):
                raise
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{{epoch:02d}}_{{val_loss:.2f}}.h5'.format()
            ),
            monitor='val_loss',
            verbose=1,
            mode='auto',
            save_best_only=False,
            period=1
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    # callback 3
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='auto',
        min_delta=0.00001,
        cooldown=0,
        min_lr=0
    ))

    # callback 4
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00001,
        patience=6,
        verbose=1,
        mode='auto'
    ))

    return callbacks


def main():
    """main script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='dataset path',
                        default=['E:/data/aug_eeg', 'E:/data/rgbimage', 'E:/data/aug_rgbimage_list',
                                 'E:/data/depthimage',
                                 'E:/data/aug_depthimage_list',
                                 'E:/data/aug_skeletontxt'])
    parser.add_argument('--gpu_list', help='gpu list', default='0')
    parser.add_argument('--epochs', help='', default=100)
    parser.add_argument('--batch_size', help='', default=1)
    parser.add_argument('--sample_num', help='', default=40)
    parser.add_argument('--learning_rate', help='', default=1e-3)
    parser.add_argument('--snapshot_path', help='', default='./snapshot')
    parser.add_argument('--tensorboard_dir', help='', default='./log')
    parser.add_argument('--is_evaluation', help='eval per epoch', default=True)
    parser.add_argument('--is_snapshot', help='resum training from a snapshot.if true,it is resum', default=False)
    args = parser.parse_args()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = args.gpu_list
    # gpu
    K.tensorflow_backend.set_session(tf.Session(config=sess_config))

    # create the generators
    train_generator, validation_generator = create_generators(args)

    # create the model
    print('creating model')
    model, training_model = create_models(
        class_num=1,
        lr=args.learning_rate,
        sample_num=args.sample_num
    )

    # print model summary
    model.summary()

    if not args.is_evaluation:
        validation_generator = None

    # create the callbacks
    callbacks = create_callbacks(
        model,
        args,
    )

    # start training
    return training_model.fit_generator(
        generator=train_generator,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator
    )


if __name__ == '__main__':
    main()
