import tensorflow as tf
import tool.tensorboard.tensor_board as tb


def MnistTestNn():
    # 1.读取数据
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # 2.建立全连接神经网络，relu作为激活函数，dropout率为0.2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    # 3.用softmax进行多分类
    predictions = model(x_train[:1]).numpy()
    tf.nn.softmax(predictions).numpy()
    # 4.定义交叉熵损失函数
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    # 4.2定义tensorBoard
    tensorboard_callback = tb.draw_board('mnist')
    #5.进行训练
    model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
    model.evaluate(x_test, y_test, verbose=2)
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])


def main():
    MnistTestNn()


if __name__ == "__main__":
    main()
