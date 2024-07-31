import tensorflow as tf

a = tf.keras.utils.to_categorical([0, 1, 2],    num_classes=3)
b = tf.keras.utils.to_categorical([-3, -2, -1], num_classes=3)

print(a)
print(b)