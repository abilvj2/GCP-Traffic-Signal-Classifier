# Import necessary libraries
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt

# Check TensorFlow version and GPU availability
print('TensorFlow version:', tf.__version__)
print('Is using GPU?', tf.test.is_gpu_available())

# Define functions

# Function to filter and shuffle data
def filter_and_shuffle_classes(x, y):
    indices_0, _ = np.where(y == 0.)
    indices_1, _ = np.where(y == 1.)
    indices_2, _ = np.where(y == 2.)

    indices = np.concatenate([indices_0, indices_1, indices_2], axis=0)
    
    x = x[indices]
    y = y[indices]
    
    count = x.shape[0]
    indices = np.random.choice(range(count), count, replace=False)
    
    x = x[indices]
    y = y[indices]
    
    y = tf.keras.utils.to_categorical(y)
    
    return x, y

# Function to show random examples
def show_random_examples(x, y, p):
    class_names = ['aeroplane', 'car', 'bird']
    indices = np.random.choice(range(x.shape[0]), 10, replace=False)
    
    x = x[indices]
    y = y[indices]
    p = p[indices]
    
    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.subplot(2, 5, 1 + i)
        plt.imshow(x[i])
        plt.xticks([])
        plt.yticks([])
        col = 'green' if np.argmax(y[i]) == np.argmax(p[i]) else 'red'
        plt.xlabel(class_names[np.argmax(p[i])], color=col)
    plt.show()

# Function to create CNN model
def create_traffic_sign_classifier_model():
    def add_conv_block(model, num_filters):
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.5))
        return model
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(32,32,3)))
    
    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)
    
    model.add(Flatten())
    model.add(Dense(3, activation='softmax')) 
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
(x_train, y_train) = filter_and_shuffle_classes(x_train, y_train)
(x_test, y_test) = filter_and_shuffle_classes(x_test, y_test)

# Visualize sample data
show_random_examples(x_train, y_train, y_train)
show_random_examples(x_test, y_test, y_test)

# Create the CNN model
traffic_sign_classifier_model = create_traffic_sign_classifier_model()
traffic_sign_classifier_model.summary()

# Train the model
history = traffic_sign_classifier_model.fit(x_train/255., y_train,
                                            validation_data=(x_test/255., y_test),
                                            epochs=10, batch_size=128,
                                            callbacks=[
                                                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3),
                                                tf.keras.callbacks.ModelCheckpoint('models/model_{val_accuracy:.3f}.h5',
                                                                                  save_best_only=True,
                                                                                  save_weights_only=False,
                                                                                  monitor='val_accuracy')
                                            ])

# Plot accuracy and validation accuracy
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(range(len(accuracy)), accuracy, label='Training')
plt.plot(range(len(val_accuracy)), val_accuracy, label='Validation')
plt.legend()
plt.show()

# Load the best model and make predictions
best_model = tf.keras.models.load_model('models/model_0.883.h5')
predictions = best_model.predict(x_test/255.)

# Show final predictions
show_random_examples(x_test, y_test, predictions)
