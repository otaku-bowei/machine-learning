import string

import matplotlib.pyplot as plt
import keras


# draw_keras_by_key 描绘训练过程中的某些学习参数
def draw_keras_by_key(model_fit_history: keras.src.callbacks.history.History, key: string):
    print("\n")
    print(model_fit_history.history[key])
    print("\n")
    plt.plot(model_fit_history.history[key], label=key)
    plt.xlabel(key)
    plt.ylabel(key + "-value")
    plt.legend()
    plt.show()

