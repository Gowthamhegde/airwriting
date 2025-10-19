#!/usr/bin/env python3
"""
Advanced Model Training for Simple Word Recognition
Focuses on training a model that can recognize simple words like CAT, DOG, etc.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Input,
    Activation, Add, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split

class AdvancedLetterTrainer:
    def __init__(self):
        self.NUM_CLASSES = 26  # A-Z
        self.IMG_SIZE = 32     # Increased from 28 for better detail
        self.BATCH_SIZE = 32   # Smaller batch for better convergence
        self.EPOCHS = 100      # More epochs with early stopping
        
        # Target words for optimization
        self.target_words = [
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT',
            'BIG', 'PIG', 'DIG', 'FIG', 'WIG', 'JIG',
            'SUN', 'RUN', 'FUN', 'GUN', 'BUN', 'NUN',
            'BOX', 'FOX', 'COX', 'SOX',
            'BED', 'RED', 'LED', 'FED',
            'TOP', 'HOP', 'MOP', 'POP', 'COP',
            'CUP', 'PUP', 'SUP',
            'BAG', 'TAG', 'RAG', 'SAG', 'WAG',
            'BUS', 'YES', 'NET', 'PET', 'SET', 'WET', 'GET', 'LET', 'MET',
            'HOT', 'POT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
            'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD',
            'BEE', 'SEE', 'TEE', 'FEE',
            'EGG', 'LEG', 'BEG', 'PEG',
            'ICE', 'NICE', 'RICE', 'MICE'
        ]
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        print(f"🎯 Training model optimized for {len(self.target_words)} simple words")
    
    def create_data_augmentation(self):
        """Create comprehensive data augmentation"""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=False,  # Don't flip letters
            fill_mode='constant',
            cval=1.0,  # White background
            brightness_range=[0.8, 1.2],
            channel_shift_range=0.1
        )
    
    def load_and_preprocess_data(self):
        """Load and preprocess EMNIST Letters dataset with advanced preprocessing"""
        print("📚 Loading EMNIST Letters dataset...")
        
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
        
        def advanced_preprocess(image, label):
            # Convert to float and normalize
            image = tf.cast(image, tf.float32) / 255.0
            
            # Resize to target size
            image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
            
            # Add slight blur to simulate air writing
            image = tf.nn.conv2d(
                tf.expand_dims(image, 0),
                tf.ones((3, 3, 1, 1)) / 9.0,
                strides=[1, 1, 1, 1],
                padding='SAME'
            )[0]
            
            # Ensure single channel
            if len(image.shape) == 3:
                image = tf.expand_dims(image[:, :, 0], -1)
            else:
                image = tf.expand_dims(image, -1)
            
            # EMNIST labels are 1-26, convert to 0-25
            label = tf.one_hot(label - 1, self.NUM_CLASSES)
            
            return image, label
        
        # Apply preprocessing
        ds_train = ds_train.map(advanced_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(advanced_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Create validation split
        train_size = int(0.85 * ds_info.splits['train'].num_examples)
        ds_train_split = ds_train.take(train_size)
        ds_val = ds_train.skip(train_size)
        
        # Batch and prefetch
        ds_train_split = ds_train_split.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_val = ds_val.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        return ds_train_split, ds_val, ds_test
    
    def create_residual_block(self, x, filters, kernel_size=3, stride=1):
        """Create a residual block for better gradient flow"""
        shortcut = x
        
        # First conv layer
        x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second conv layer
        x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add shortcut
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        return x
    
    def create_advanced_model(self):
        """Create advanced CNN with residual connections"""
        inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 1))
        
        # Initial conv layer
        x = Conv2D(32, 7, strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        
        # Residual blocks
        x = self.create_residual_block(x, 32)
        x = self.create_residual_block(x, 32)
        
        x = self.create_residual_block(x, 64, stride=2)
        x = self.create_residual_block(x, 64)
        
        x = self.create_residual_block(x, 128, stride=2)
        x = self.create_residual_block(x, 128)
        
        # Global average pooling
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers with dropout
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(self.NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model
    
    def create_learning_rate_schedule(self):
        """Create learning rate schedule"""
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 30:
                return lr * 0.9
            elif epoch < 60:
                return lr * 0.8
            else:
                return lr * 0.7
        
        return LearningRateScheduler(scheduler)
    
    def train_model(self):
        """Train the advanced model"""
        print("🚀 Starting advanced model training...")
        
        # Load data
        ds_train, ds_val, ds_test = self.load_and_preprocess_data()
        
        # Create model
        model = self.create_advanced_model()
        
        # Compile with advanced optimizer
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Print model summary
        print("🏗️  Model Architecture:")
        model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'models/advanced_letter_recognition.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            self.create_learning_rate_schedule()
        ]
        
        # Train model
        print("🎓 Training started...")
        history = model.fit(
            ds_train,
            epochs=self.EPOCHS,
            validation_data=ds_val,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = model.evaluate(ds_test, verbose=1)
        print(f"Test Accuracy: {test_results[1]:.4f}")
        print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")
        
        # Save final model
        model.save("models/letter_recognition_advanced.h5")
        print("💾 Advanced model saved as 'models/letter_recognition_advanced.h5'")
        
        # Plot training history
        self.plot_training_history(history)
        
        # Test on target words
        self.test_target_words(model)
        
        return model, history
    
    def plot_training_history(self, history):
        """Plot comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-3 Accuracy
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Training Top-3', linewidth=2)
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation Top-3', linewidth=2)
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig('models/advanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 Training history plot saved as 'models/advanced_training_history.png'")
    
    def test_target_words(self, model):
        """Test model performance on target words"""
        print("\n🎯 Testing performance on target words...")
        
        # Create simple test images for each letter
        def create_letter_image(letter):
            """Create a simple synthetic letter image"""
            img = np.ones((self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.float32)
            
            # Use OpenCV to draw the letter (simplified)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
            
            # Center the text
            x = (self.IMG_SIZE - text_width) // 2
            y = (self.IMG_SIZE + text_height) // 2
            
            cv2.putText(img, letter, (x, y), font, font_scale, (0,), thickness)
            
            return img
        
        # Test a few sample words
        sample_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED']
        
        for word in sample_words:
            print(f"\n  Testing word: {word}")
            word_accuracy = []
            
            for letter in word:
                # Create test image
                test_img = create_letter_image(letter)
                test_img = np.expand_dims(test_img, 0)  # Add batch dimension
                
                # Predict
                prediction = model.predict(test_img, verbose=0)
                predicted_idx = np.argmax(prediction)
                predicted_letter = chr(predicted_idx + ord('A'))
                confidence = np.max(prediction)
                
                is_correct = predicted_letter == letter
                word_accuracy.append(is_correct)
                
                status = "✅" if is_correct else "❌"
                print(f"    {letter} -> {predicted_letter} ({confidence:.3f}) {status}")
            
            word_acc = sum(word_accuracy) / len(word_accuracy) * 100
            print(f"  Word accuracy: {word_acc:.1f}%")
    
    def create_optimized_word_dictionary(self):
        """Create optimized dictionary for target words"""
        print("📚 Creating optimized word dictionary...")
        
        # Calculate letter frequencies in target words
        letter_freq = {}
        for word in self.target_words:
            for letter in word:
                letter_freq[letter] = letter_freq.get(letter, 0) + 1
        
        # Sort by frequency
        sorted_letters = sorted(letter_freq.items(), key=lambda x: x[1], reverse=True)
        
        print("📊 Letter frequency in target words:")
        for letter, freq in sorted_letters[:10]:
            print(f"   {letter}: {freq} times")
        
        # Save dictionary
        dictionary_data = {
            'target_words': self.target_words,
            'letter_frequencies': letter_freq,
            'total_words': len(self.target_words)
        }
        
        import json
        with open('models/word_dictionary.json', 'w') as f:
            json.dump(dictionary_data, f, indent=2)
        
        print("💾 Word dictionary saved as 'models/word_dictionary.json'")

def main():
    """Main training function"""
    print("🧠 ADVANCED LETTER RECOGNITION MODEL TRAINING")
    print("=" * 60)
    print("🎯 Optimized for simple words like CAT, DOG, SUN, etc.")
    print("=" * 60)
    
    trainer = AdvancedLetterTrainer()
    
    try:
        # Create optimized dictionary
        trainer.create_optimized_word_dictionary()
        
        # Train the model
        model, history = trainer.train_model()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Files created:")
        print("   • models/letter_recognition_advanced.h5 (Best model)")
        print("   • models/advanced_letter_recognition.h5 (Checkpoint)")
        print("   • models/advanced_training_history.png (Training plots)")
        print("   • models/word_dictionary.json (Optimized dictionary)")
        print("\n🚀 Ready to use with enhanced air writing system!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()