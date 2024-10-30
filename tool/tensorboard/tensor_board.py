import datetime
import os
import string

import keras.src.callbacks
import tensorflow as tf


# draw_board 响应一个call_back的路径
def draw_board(path: string) -> keras.src.callbacks.TensorBoard:
    current_file_abs_path = os.path.realpath(__file__)
    current_file_abs_path = os.path.dirname(current_file_abs_path)
    log_dir = current_file_abs_path + "/" + path + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_callback
