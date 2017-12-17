from model import Tower
from utils import model_property
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as digits


class UserModel(Tower):

    @model_property
    def inference(self):
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
                                
            model = slim.batch_norm(x, is_training=self.is_training, center=True, scale=True)
            model = slim.conv2d(model, 3, [1, 1], padding='SAME',  scope='conv0')
            
            model = slim.conv2d(model, 16, [5, 5], padding='SAME', scope='conv1')
            model = slim.max_pool2d(model, [2, 2], padding='SAME', scope='pool1')
            model = slim.dropout(model, 0.7, is_training=self.is_training, scope='do1')
            model = slim.batch_norm(model, is_training=self.is_training, center=True, scale=True)
            
            model = slim.conv2d(model, 32, [5, 5], padding='SAME', scope='conv2')
            model = slim.max_pool2d(model, [2, 2], padding='SAME', scope='pool2')
            model = slim.dropout(model, 0.7, is_training=self.is_training, scope='do2')
            model = slim.batch_norm(model, is_training=self.is_training, center=True, scale=True)
            
            model = slim.flatten(model)
            model = slim.fully_connected(model, 2048, scope='fc1')
            model = slim.dropout(model, 0.7, is_training=self.is_training, scope='do3')
            model = slim.fully_connected(model, self.nclasses, activation_fn=None, scope='fc2')
            return model

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
