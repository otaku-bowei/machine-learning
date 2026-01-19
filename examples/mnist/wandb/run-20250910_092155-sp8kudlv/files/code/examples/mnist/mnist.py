import tensorflow as tf
import wandb
from datetime import datetime

API_KEY = "local-0e045fef2630a4f59fdf629d25206dfc2c6af98e"

def MnistTestNn():
    wandb.login(key=API_KEY)
    # 初始化wandb
    wandb.init(
        project="mnist-neural-network",
        name=f"mnist-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "model_type": "Sequential",
            "input_shape": (28, 28),
            "hidden_units": 128,
            "dropout_rate": 0.2,
            "optimizer": "adam",
            "loss_function": "SparseCategoricalCrossentropy",
            "metrics": ["accuracy"],
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        tags=["mnist", "neural-network", "tensorflow", "classification"]
    )

    # 1.读取数据
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 记录数据信息
    wandb.log({
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "input_shape": x_train.shape[1:],
        "num_classes": len(set(y_train))
    })

    # 2.建立全连接神经网络
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
        tf.keras.layers.Dense(128, activation='relu', name="dense_128"),
        tf.keras.layers.Dropout(0.2, name="dropout"),
        tf.keras.layers.Dense(10, name="output")
    ])

    # 3.定义损失函数和编译模型
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=['accuracy']
    )

    # 记录模型结构
    wandb.log({"model_summary": model.summary()})

    # 4.创建wandb回调
    wandb_callback = wandb.keras.WandbCallback(
        monitor="val_accuracy",
        save_model=True,
        save_graph=True,
        log_weights=True,
        log_gradients=True,
        training_data=(x_train[:1000], y_train[:1000]),
        validation_data=(x_test[:200], y_test[:200]),
        log_evaluation=True,
        compute_importance=True
    )

    # 5.训练模型
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[wandb_callback],
        verbose=1
    )

    # 6.评估模型
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

    # 记录最终指标
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "final_training_accuracy": history.history['accuracy'][-1],
        "final_validation_accuracy": history.history['val_accuracy'][-1]
    })

    # 7.创建概率模型并预测
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    # 记录预测样本
    predictions = probability_model(x_test[:5])
    predicted_classes = tf.argmax(predictions, axis=1).numpy()

    # 记录预测结果
    for i in range(5):
        wandb.log({
            f"sample_{i}": {
                "true_label": int(y_test[i]),
                "predicted_label": int(predicted_classes[i]),
                "confidence": float(tf.reduce_max(predictions[i]))
            }
        })

    # 保存模型
    model.save('mnist_model.h5')
    wandb.save('mnist_model.h5')

    wandb.finish()
    return model


def main():
    MnistTestNn()


if __name__ == "__main__":
    main()