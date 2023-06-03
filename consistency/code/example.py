#Filename:	example.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 08 Agu 2022 03:49:23 

import tensorflow as tf 
import numpy as np

from consistency import IterativeSearch
from consistency import PGDsL2
from consistency import StableNeighborSearch

from utils import load_dataset
from utils import invalidation

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def dnn(input_shape, n_classes=2):
    x = tf.keras.Input(input_shape)
    y = tf.keras.layers.Dense(128)(x)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Dense(128)(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Dense(n_classes)(y)
    y = tf.keras.layers.Activation('softmax')(y)
    return tf.keras.models.Model(x, y)

def train_dnn(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    model.evaluate(X_test, y_test, batch_size=batch_size)
    return model

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test), n_classes = load_dataset('Seizure', path_to_data_dir='dataset/data')

    baseline_model = dnn(X_train.shape[1:], n_classes=n_classes)
    baseline_model = train_dnn(baseline_model, X_train, y_train, X_test, y_test, batch_size=256)

    model_1 = dnn(X_train.shape[1:], n_classes=n_classes)
    model_1 = train_dnn(baseline_model, X_train, y_train, X_test, y_test, batch_size=256)

    sns_fn = StableNeighborSearch(baseline_model,
                     clamp=[X_train.min(), X_train.max()],
                     num_classes=2,
                     sns_eps=0.1,
                     sns_nb_iters=100,
                     sns_eps_iter=1.e-3,
                     n_interpolations=20)
    
    '''
    L1_iter_search = IterativeSearch(baseline_model,
                                clamp=[X_train.min(), X_train.max()],
                                num_classes=2,
                                eps=0.3,
                                nb_iters=40,
                                eps_iter=0.01,
                                norm=1,
                                sns_fn=sns_fn)

    l1_cf, pred_cf, is_valid = L1_iter_search(X_test[:128])
    iv = invalidation(l1_cf,
                    np.argmax(baseline_model.predict(X_test[:128]), axis=1),
                    model_1,
                    affinity_set=[[0], [1]])

    print(f"Invalidation Rate: {iv}")

    L2_iter_search = IterativeSearch(baseline_model,
                                clamp=[X_train.min(), X_train.max()],
                                num_classes=2,
                                eps=0.3,
                                nb_iters=40,
                                eps_iter=0.01,
                                norm=2,
                                sns_fn=sns_fn)
    l2_cf, pred_cf, is_valid = L2_iter_search(X_test[:128])

    iv  = invalidation(l2_cf,
                    np.argmax(baseline_model.predict(X_test[:128]), axis=1),
                    model_1,
                    affinity_set=[[0], [1]])

    print(f"Invalidation Rate: {iv}")
    '''

    pgd_iter_search = PGDsL2(baseline_model,
                        clamp=[X_train.min(), X_train.max()],
                        num_classes=2,
                        eps=2.0,
                        nb_iters=100,
                        eps_iter=0.04,
                        sns_fn=sns_fn)
    pgd_cf, pred_cf, is_valid = pgd_iter_search(X_test[:128], num_interpolations=10, batch_size=64)

    iv = invalidation(pgd_cf,
                    np.argmax(baseline_model.predict(X_test[:128]), axis=1),
                    model_1,
                    batch_size=32,
                    affinity_set=[[0], [1]])

    print(f"Invalidation Rate: {iv}")
