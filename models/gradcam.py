import tensorflow as tf
import numpy as np
import cv2

def generate_heatmap(model, img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.layers[-1].output, model.layers[-3].output]
    )

    with tf.GradientTape() as tape:
        preds, conv_outputs = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()