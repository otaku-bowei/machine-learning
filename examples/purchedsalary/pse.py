import numpy
import tensorflow as tf

def PurchedSalaryTestNn():
    #1.读取数据
    data = numpy.loadtxt("./Social_Network_Ads_train.csv", delimiter=',')
    trainData = tf.reshape(tf.convert_to_tensor(data[:, 1:-1]), [381, 1, 3])
    trainLabel = tf.reshape(tf.convert_to_tensor(data[:, -1:]), [381, 1, 1])
    mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    #2.建立全连接神经网络，relu作为激活函数，dropout率为0.2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(1, 3)),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2)
    ])
    #3.用softmax进行多分类
    predictions = model(trainData[:1]).numpy()
    tf.nn.sigmoid(predictions).numpy()
    #4.定义交叉熵损失函数
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(trainLabel[:1], predictions).numpy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    #5.进行训练
    model.fit(trainData, trainLabel, epochs=10)
    model.evaluate(trainData, trainLabel, verbose=2)
    # probability_model = tf.keras.Sequential([
    #     model,
    #     tf.keras.layers.Softmax()
    # ])
    # probability_model(x_test[:5])


def main():
    PurchedSalaryTestNn()


if __name__ == "__main__":
    main()