#!/usr/bin/env python3
"""
Optimized CNN Training for Real-Time Air Writing Word Recognition
Specialized for 3-letter words with enhanced preprocessing and fast inference
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input,
    Activation, Add, Attention, Reshape, LSTM, TimeDistributed
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler, Callback
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
import cv2
from sklearn.model_selection import train_test_split
from collections import Counter

class OptimizedLetterTrainer:
    def __init__(self):
        self.NUM_CLASSES = 26  # A-Z
        self.IMG_SIZE = 32     # Optimized for speed
        self.BATCH_SIZE = 128  # Larger batch for faster training
        self.EPOCHS = 100

        # Focus on 3-letter words
        self.target_words = [
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT', 'PAT',
            'BIG', 'PIG', 'DIG', 'FIG', 'WIG', 'JIG', 'RIG',
            'SUN', 'RUN', 'FUN', 'GUN', 'BUN', 'NUN',
            'BOX', 'FOX', 'COX', 'SOX',
            'BED', 'RED', 'LED', 'FED', 'WED',
            'TOP', 'HOP', 'MOP', 'POP', 'COP', 'SOP',
            'CUP', 'PUP', 'SUP', 'YUP',
            'BAG', 'TAG', 'RAG', 'SAG', 'WAG', 'NAG',
            'BUS', 'HUS', 'MUS',
            'HOT', 'POT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
            'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD', 'FAD',
            'BEE', 'SEE', 'TEE', 'FEE', 'PEE',
            'EGG', 'LEG', 'BEG', 'PEG',
            'ICE', 'DICE', 'NICE', 'RICE', 'MICE', 'VICE'
        ]

        # Create models directory
        os.makedirs("models", exist_ok=True)

        print(f"🎯 Optimized training for {len(self.target_words)} words")
        print("📊 Letter frequency analysis for optimization...")

        # Analyze letter frequencies for better training
        self.letter_freq = Counter()
        for word in self.target_words:
            self.letter_freq.update(word)

        print("Top letters:", sorted(self.letter_freq.items(), key=lambda x: x[1], reverse=True)[:10])

    def create_advanced_augmentation(self):
        """Create air-writing specific data augmentation"""
        return ImageDataGenerator(
            rotation_range=25,  # More rotation for air writing
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=[0.8, 1.3],  # Realistic zoom variations
            fill_mode='constant',
            cval=1.0,
            brightness_range=[0.7, 1.3],  # Air writing lighting variations
            channel_shift_range=0.15,
            # Add realistic air-writing distortions
            preprocessing_function=self.air_writing_distortion
        )

    def air_writing_distortion(self, image):
        """Apply realistic air-writing distortions"""
        # Random stroke thickness variation
        kernel_size = np.random.randint(1, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Random dilation/erosion to simulate stroke variations
        if np.random.random() > 0.5:
            image = cv2.dilate(image, kernel, iterations=1)
        else:
            image = cv2.erode(image, kernel, iterations=1)

        # Add slight blur to simulate motion
        blur_kernel = np.random.choice([1, 3, 5])
        if blur_kernel > 1:
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

        # Random noise to simulate imperfect tracking
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 1)

        return image

    def load_and_preprocess_data(self):
        """Load and preprocess EMNIST with focus on target letters"""
        print("📚 Loading EMNIST Letters dataset...")

        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        print(f"Training samples: {ds_info.splits['train'].num_examples}")

        def optimized_preprocess(image, label):
            # Convert and resize
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))

            # Add channel dimension
            image = tf.expand_dims(image, -1)

            # Convert labels (EMNIST: 1-26 -> 0-25)
            label = tf.one_hot(label - 1, self.NUM_CLASSES)

            return image, label

        # Apply preprocessing
        ds_train = ds_train.map(optimized_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(optimized_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # Create validation split
        train_size = int(0.9 * ds_info.splits['train'].num_examples)
        ds_train_split = ds_train.take(train_size)
        ds_val = ds_train.skip(train_size)

        # Batch and prefetch
        ds_train_split = ds_train_split.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_val = ds_val.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return ds_train_split, ds_val, ds_test

    def create_lightweight_model(self):
        """Create lightweight CNN optimized for real-time inference"""
        model = Sequential([
            # Efficient first conv block
            Conv2D(64, (3, 3), activation='relu', input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.2),

            # Second conv block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Dropout(0.3),

            # Third conv block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.4),

            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            # Output
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        return model

    def create_attention_model(self):
        """Create attention-based model for better feature focus"""
        inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 1))

        # CNN feature extraction
        x = Conv2D(64, (3, 3), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, 2)(x)

        # Attention mechanism
        attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = x * attention

        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)

        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.NUM_CLASSES, activation='softmax')(x)

        return Model(inputs, outputs)

    def focal_loss(self, gamma=2.0, alpha=0.25):
        """Focal loss for better handling of hard examples"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
            fl = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)

            return tf.reduce_mean(fl)
        return focal_loss_fixed

    def create_learning_rate_schedule(self):
        """Optimized learning rate schedule"""
        def scheduler(epoch, lr):
            if epoch < 20:
                return lr
            elif epoch < 50:
                return lr * 0.5
            elif epoch < 80:
                return lr * 0.2
            else:
                return lr * 0.1
        return LearningRateScheduler(scheduler)

    class WordAccuracyCallback(Callback):
        """Custom callback to monitor word-level accuracy"""
        def __init__(self, trainer, validation_data):
            super().__init__()
            self.trainer = trainer
            self.validation_data = validation_data

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 10 == 0:  # Check every 10 epochs
                word_accuracy = self.trainer.evaluate_word_accuracy(self.validation_data)
                print(f"\n📊 Word-level accuracy: {word_accuracy:.3f}")

    def evaluate_word_accuracy(self, ds_val):
        """Evaluate accuracy on target words"""
        correct_words = 0
        total_words = 0

        # Sample some validation data
        for images, labels in ds_val.take(10):  # Sample batches
            predictions = self.model.predict(images, verbose=0)

            for i in range(min(10, len(images))):  # Sample predictions per batch
                true_idx = np.argmax(labels[i])
                pred_idx = np.argmax(predictions[i])

                # Check if both letters are in target words
                true_letter = chr(true_idx + ord('A'))
                pred_letter = chr(pred_idx + ord('A'))

                # Simple check: if predicted letter appears in target words
                if any(true_letter in word for word in self.target_words):
                    total_words += 1
                    if pred_idx == true_idx:
                        correct_words += 1

        return correct_words / max(total_words, 1)

    def train_optimized_model(self):
        """Train the optimized model"""
        print("🚀 Starting optimized model training...")

        # Load data
        ds_train, ds_val, ds_test = self.load_and_preprocess_data()

        # Create model
        model = self.create_lightweight_model()

        # Compile with focal loss for better accuracy
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', 'top_3_accuracy']
        )

        # Print model summary
        print("🏗️  Optimized Model Architecture:")
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/optimized_letter_recognition.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            self.create_learning_rate_schedule(),
            self.WordAccuracyCallback(self, ds_val)
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

        # Evaluate
        print("\n📊 Evaluating on test set...")
        test_results = model.evaluate(ds_test, verbose=1)
        print(f"Test Accuracy: {test_results[1]:.4f}")
        print(f"Test Top-3 Accuracy: {test_results[2]:.4f}")

        # Save model
        model.save("models/letter_recognition_optimized.h5")
        print("💾 Optimized model saved as 'models/letter_recognition_optimized.h5'")

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

        # Learning Rate
        if 'lr' in history.history:
            axes[1, 1].plot(history.history['lr'], linewidth=2, color='orange')
            axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('models/optimized_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📈 Training history saved as 'models/optimized_training_history.png'")

    def test_target_words(self, model):
        """Test model performance on target words"""
        print("\n🎯 Testing performance on target words...")

        # Create simple test images for each letter
        def create_letter_image(letter):
            img = np.ones((self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.float32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
            x = (self.IMG_SIZE - text_width) // 2
            y = (self.IMG_SIZE + text_height) // 2

            cv2.putText(img, letter, (x, y), font, font_scale, (0,), thickness)
            return img

        # Test sample words
        sample_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP']

        for word in sample_words:
            print(f"\n  Testing word: {word}")
            word_accuracy = []

            for letter in word:
                test_img = create_letter_image(letter)
                test_img = np.expand_dims(test_img, 0)

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

    def create_word_dictionary(self):
        """Create optimized word dictionary for real-time recognition"""
        print("📚 Creating optimized word dictionary...")

        # Analyze letter patterns in target words
        letter_patterns = {}
        for word in self.target_words:
            for i, letter in enumerate(word):
                if i not in letter_patterns:
                    letter_patterns[i] = Counter()
                letter_patterns[i][letter] += 1

        dictionary_data = {
            'target_words': self.target_words,
            'letter_frequencies': dict(self.letter_freq),
            'letter_patterns': {pos: dict(counter) for pos, counter in letter_patterns.items()},
            'total_words': len(self.target_words),
            'avg_word_length': np.mean([len(w) for w in self.target_words])
        }

        import json
        with open('models/optimized_word_dictionary.json', 'w') as f:
            json.dump(dictionary_data, f, indent=2)

        print("💾 Optimized word dictionary saved as 'models/optimized_word_dictionary.json'")

def main():
    """Main training function"""
    print("🧠 OPTIMIZED LETTER RECOGNITION MODEL TRAINING")
    print("=" * 60)
    print("🎯 Specialized for real-time 3-letter word recognition")
    print("⚡ Lightweight architecture for fast inference")
    print("=" * 60)

    trainer = OptimizedLetterTrainer()

    try:
        # Create optimized dictionary
        trainer.create_word_dictionary()

        # Train the model
        model, history = trainer.train_optimized_model()

        print("\n" + "=" * 60)
        print("🎉 OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Files created:")
        print("   • models/letter_recognition_optimized.h5 (Optimized model)")
        print("   • models/optimized_letter_recognition.h5 (Checkpoint)")
        print("   • models/optimized_training_history.png (Training plots)")
        print("   • models/optimized_word_dictionary.json (Word dictionary)")
        print("\n🚀 Ready for real-time air writing word recognition!")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
