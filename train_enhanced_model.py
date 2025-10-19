#!/usr/bin/env python3
"""
Enhanced CNN Training Script for Air Writing Recognition
Trains a more robust model with data augmentation and better architecture
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class EnhancedLetterRecognitionTrainer:
    def __init__(self):
        self.NUM_CLASSES = 26  # A-Z
        self.IMG_SIZE = 28
        self.BATCH_SIZE = 64
        self.EPOCHS = 50
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess EMNIST Letters dataset"""
        print("Loading EMNIST Letters dataset...")
        
        # Load dataset
        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        
        print(f"Training samples: {ds_info.splits['train'].num_examples}")
        print(f"Test samples: {ds_info.splits['test'].num_examples}")
        
        def preprocess(image, label):
            # Resize and normalize
            image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
            image = tf.cast(image, tf.float32) / 255.0
            
            # EMNIST labels are 1-26, subtract 1 for 0-25
            label = tf.one_hot(label - 1, self.NUM_CLASSES)
            return image, label
        
        # Apply preprocessing
        ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Create validation split from training data
        train_size = int(0.8 * ds_info.splits['train'].num_examples)
        ds_train_split = ds_train.take(train_size)
        ds_val = ds_train.skip(train_size)
        
        # Batch and prefetch
        ds_train_split = ds_train_split.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_val = ds_val.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        return ds_train_split, ds_val, ds_test
    
    def create_enhanced_model(self):
        """Create enhanced CNN architecture"""
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            # Global average pooling instead of flatten
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(self.NUM_CLASSES, activation='softmax')
        ])
        
        return model
    
    def create_data_augmentation(self):
        """Create data augmentation generator with enhanced parameters for small letters"""
        datagen = ImageDataGenerator(
            rotation_range=20,  # Increased for better letter variations
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            fill_mode='constant',
            cval=1.0,  # White background
            brightness_range=[0.8, 1.2],  # Add brightness variation
            channel_shift_range=0.1  # Slight channel shifts
        )
        return datagen
    
    def train_model(self):
        """Train the enhanced model"""
        print("Starting enhanced model training...")
        
        # Load data
        ds_train, ds_val, ds_test = self.load_and_preprocess_data()
        
        # Create model
        model = self.create_enhanced_model()
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Print model summary
        model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/best_letter_recognition.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            ds_train,
            epochs=self.EPOCHS,
            validation_data=ds_val,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = model.evaluate(ds_test, verbose=1)
        print(f"Test Accuracy: {test_results[1]:.4f}")
        print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
        
        # Save final model
        model.save("models/letter_recognition_enhanced.h5")
        print("Enhanced model saved as 'models/letter_recognition_enhanced.h5'")
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Training history plot saved as 'models/training_history.png'")
    
    def test_model_predictions(self, model, ds_test):
        """Test model predictions on sample data"""
        print("\nTesting model predictions...")
        
        # Get a batch of test data
        for images, labels in ds_test.take(1):
            predictions = model.predict(images[:10])
            
            # Display results
            fig, axes = plt.subplots(2, 5, figsize=(12, 6))
            axes = axes.ravel()
            
            for i in range(10):
                # Display image
                axes[i].imshow(images[i].numpy().squeeze(), cmap='gray')
                
                # Get true and predicted labels
                true_label = chr(np.argmax(labels[i]) + ord('A'))
                pred_label = chr(np.argmax(predictions[i]) + ord('A'))
                confidence = np.max(predictions[i])
                
                # Set title
                color = 'green' if true_label == pred_label else 'red'
                axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                                color=color, fontsize=10)
                axes[i].axis('off')
            
            plt.tight_layout()
            plt.savefig('models/sample_predictions.png', dpi=300, bbox_inches='tight')
            plt.show()
            break
        
        print("Sample predictions saved as 'models/sample_predictions.png'")

def main():
    """Main training function"""
    print("Enhanced Letter Recognition Model Training")
    print("=" * 50)
    
    trainer = EnhancedLetterRecognitionTrainer()
    
    try:
        # Train the model
        model, history = trainer.train_model()
        
        # Test predictions
        _, _, ds_test = trainer.load_and_preprocess_data()
        trainer.test_model_predictions(model, ds_test)
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print("Models saved in 'models/' directory")
        print("Use 'letter_recognition_enhanced.h5' for best performance")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()