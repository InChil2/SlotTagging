# -*- coding: utf-8 -*-

import os
import json

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K

class KorBertLayer(tf.keras.layers.Layer):
    def __init__(self, bert_path="./bert-module",
                 n_tune_layers=10, trainable=True, **kwargs):

        self.trainable = trainable
        self.n_tune_layers = n_tune_layers
        self.bert_path = bert_path
        self.output_size = 768

        super(KorBertLayer, self).__init__(**kwargs)
        print('init ok')

    def build(self, input_shape):
        self.bert = hub.Module(self.build_abspath(self.bert_path),
                               trainable=self.trainable,
                               name=f"{self.name}_module")

        # Remove unused layers
        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars
                          if not "/cls/" in var.name]

        trainable_layers = []
        # Select how many layers to fine tune
        for i in range(self.n_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [ var for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(KorBertLayer, self).build(input_shape)


    def build_abspath(self, path):
        return os.path.abspath(path)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature='tokens',
                           as_dict=True)
        return result['sequence_output']
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_size)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_tune_layers': self.n_tune_layers,
            'trainable': self.trainable,
            'output_size': self.output_size,
            'bert_path': self.bert_path,
        })
        return config

