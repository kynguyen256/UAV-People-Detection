import tensorflow as tf
from tensorflow.keras import layers, models

def setup_device():
    """
    Configures TensorFlow to use a GPU if available.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only allocate memory as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU detected. Using CPU.")

def create_model(input_shape=(224, 224, 3), num_classes=1):
    """
    Creates a ResNet50-based model for object detection.
    
    Parameters:
    - input_shape: The shape of the input images.
    - num_classes: The number of classes (for 'humans', it's 1).
    
    Returns:
    - model: A Keras Model instance.
    """
    # Load the ResNet50 model without the top classification layers
    base_model = tf.keras.applications.ResNet50(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False

    # Add custom layers on top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer for bounding box regression (4 values: x, y, width, height)
    bbox_output = layers.Dense(4, name='bbox_output')(x)
    
    # Output layer for classification (sigmoid activation for binary classification)
    class_output = layers.Dense(num_classes, activation='sigmoid', name='class_output')(x)
    
    # Define the model with two outputs
    model = models.Model(inputs=base_model.input, outputs=[bbox_output, class_output])
    
    return model

# Call setup_device() to configure GPU usage before creating the model
if __name__ == "__main__":
    setup_device()
    model = create_model()
    print("Model created successfully.")
