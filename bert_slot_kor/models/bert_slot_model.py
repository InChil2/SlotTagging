# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Input, Dense, TimeDistributed

from models.korbert_layer import KorBertLayer


class BertSlotModel:

    def __init__(self, slots_num, bert_hub_path,
                 sess, num_bert_fine_tune_layers=10, is_training=True):
        self.slots_num = slots_num
        self.bert_hub_path = bert_hub_path
        self.num_bert_fine_tune_layers = num_bert_fine_tune_layers
        self.is_training = is_training
 
        self.model_params = {
                'slots_num': slots_num,
                'bert_hub_path': bert_hub_path,
                'num_bert_fine_tune_layers': num_bert_fine_tune_layers
                }
        
        self.build_model()
        self.compile_model()
        self.initialize_vars(sess)
        
        
    def compile_model(self):
        
        optimizer = tf.keras.optimizers.Adam(lr=5e-5)#0.001)

        losses = {
        	'time_distributed': 'sparse_categorical_crossentropy',
        }

        self.model.compile(optimizer=optimizer, loss=losses, metrics=losses)
        self.model.summary()
        

    def build_model(self):
        in_id = Input(shape=(None,), dtype=tf.int32, name='input_ids')
        in_mask = Input(shape=(None,), dtype=tf.int32, name='input_mask')
        in_segment = Input(shape=(None,), dtype=tf.int32, name='segment_ids')

        bert_inputs = [in_id, in_mask, in_segment] # tf.keras layers

        print('bert inputs :', bert_inputs)

        bert_sequence_output = KorBertLayer(
            n_tune_layers=self.num_bert_fine_tune_layers,
            bert_path = self.bert_hub_path,
            name='KorBertLayer')(bert_inputs)
    
        slots_output = TimeDistributed(
            Dense(self.slots_num, activation='softmax')
        )(bert_sequence_output)

        print('slots output :', slots_output.shape)

        self.model = Model(inputs=bert_inputs, outputs=slots_output)

        
    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=64):
        """
        X: batch of [input_ids, input_mask, segment_ids]
        """

        X = (X[0], X[1], X[2])
        if validation_data is not None:
            X_val, Y_val = validation_data
            validation_data = ((X_val[0], X_val[1], X_val[2]), Y_val)

        history = self.model.fit(X, Y, validation_data=validation_data,
                                 epochs=epochs, batch_size=batch_size)
        self.visualize_log(history.history, 'loss')
        
    def initialize_vars(self, sess):
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        K.set_session(sess)
        
    def predict_slots(self, x, slots_to_array, remove_start_end=True):

        input_ids = x[0]
        y_slots = self.model.predict(x)

        slots = slots_to_array.inverse_transform(y_slots, input_ids)

        def notPAD(element):
            if element == '<PAD>':
                return False
            else:
                return True
        slots = [list(filter(notPAD, x)) for x in slots]
        y_slots = np.array(y_slots)
        
        if remove_start_end:
            slots = [x[1:-1] for x in slots]
            y_slots = np.array([[x for x in y_slots[i][1:(len(slots[i])+1)]]
                                for i in range(y_slots.shape[0])])

        slots_score = [[None]*len(slots[i]) for i in range(y_slots.shape[0])]
        temp = list(slots_to_array.label_encoder.classes_)
        for i, slot in enumerate(slots):
            for j, tag in enumerate(slot):
                slots_score[i][j] = y_slots[i][j][temp.index(tag)]

        return slots, slots_score

    def save(self, model_path):
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save(os.path.join(model_path, 'bert_slot_model.h5'))
        
    def load(load_folder_path, sess): # load for inference or model evaluation
        params_path = os.path.join(load_folder_path, 'params.json')
        with open(params_path, 'r') as json_file:
            model_params = json.load(json_file)
            
        slots_num = model_params['slots_num'] 
        bert_hub_path = model_params['bert_hub_path']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']
            
        new_model = BertSlotModel(slots_num, bert_hub_path, sess,
                                  num_bert_fine_tune_layers, is_training=False)
        h5_path = os.path.join(load_folder_path,'bert_slot_model.h5')
        new_model.model.load_weights(h5_path)
        return new_model

    def visualize_log(self, history_dic, metric_name):
        print("visualizing...")
        plt.plot(history_dic["loss"])
        plt.plot(history_dic["val_loss"])
        plt.ylim([0, 0.001])
        plt.title("LOSS")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["train", "validation"])
        plt.savefig("/content/drive/MyDrive/Slot_tagging_project/data/save_models/loss.png", dpi=300)

        
