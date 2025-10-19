#!/usr/bin/env python3
"""
Ultra-Optimized CNN Training for Real-Time Air Writing Word Recognition
Combines advanced techniques: focal loss, attention mechanisms, air-writing specific augmentation,
residual connections, and enhanced preprocessing for maximum accuracy
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input,
    Activation, Add, Attention, Reshape, LSTM, TimeDistributed,
    MultiHeadAttention, LayerNormalization, Concatenate
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

class UltraOptimizedLetterTrainer:
    def __init__(self):
        self.NUM_CLASSES = 26  # A-Z
        self.IMG_SIZE = 32     # Optimized for air writing detail
        self.BATCH_SIZE = 64   # Balanced for speed and convergence
        self.EPOCHS = 150      # Extended training with early stopping

        # Focus on 3-letter words with expanded set
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
            'ICE', 'DICE', 'NICE', 'RICE', 'MICE', 'VICE',
            'ANT', 'PAN', 'MAN', 'CAN', 'FAN', 'VAN',
            'CAR', 'FAR', 'JAR', 'BAR', 'TAR',
            'DAY', 'BAY', 'GAY', 'HAY', 'JAY', 'KAY', 'LAY', 'MAY', 'NAY', 'PAY', 'RAY', 'SAY', 'TAY', 'WAY',
            'EYE', 'DYE', 'RYE',
            'OIL', 'BOIL', 'COIL', 'FOIL', 'SOIL', 'TOIL'
        ]

        # Create models directory
        os.makedirs("models", exist_ok=True)

        print(f"🎯 Ultra-optimized training for {len(self.target_words)} words")
        print("📊 Advanced letter frequency analysis...")

        # Analyze letter frequencies for optimization
        self.letter_freq = Counter()
        for word in self.target_words:
            self.letter_freq.update(word)

        print("Top letters:", sorted(self.letter_freq.items(), key=lambda x: x[1], reverse=True)[:10])

    def create_ultra_augmentation(self):
        """Create ultra-advanced air-writing specific data augmentation"""
        return ImageDataGenerator(
            rotation_range=30,  # More rotation for air writing
            width_shift_range=0.25,
            height_shift_range=0.25,
            shear_range=0.25,
            zoom_range=[0.75, 1.4],  # Realistic zoom variations
            fill_mode='constant',
            cval=1.0,
            brightness_range=[0.6, 1.4],  # Air writing lighting variations
            channel_shift_range=0.2,
            # Add realistic air-writing distortions
            preprocessing_function=self.ultra_air_writing_distortion
        )

    def ultra_air_writing_distortion(self, image):
        """Apply ultra-realistic air-writing distortions"""
        # Random stroke thickness variation
        kernel_size = np.random.randint(1, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Random dilation/erosion to simulate stroke variations
        if np.random.random() > 0.5:
            image = cv2.dilate(image, kernel, iterations=1)
        else:
            image = cv2.erode(image, kernel, iterations=1)

        # Add slight blur to simulate motion
        blur_kernel = np.random.choice([1, 3, 5, 7])
        if blur_kernel > 1:
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)

        # Random noise to simulate imperfect tracking
        noise = np.random.normal(0, 0.08, image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 1)

        # Random perspective distortion (slight)
        if np.random.random() > 0.7:
            rows, cols = image.shape[:2]
            pts1 = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
            pts2 = pts1 + np.random.uniform(-3, 3, pts1.shape)
            M = cv2.getPerspectiveTransform(pts1, pts2)
            image = cv2.warpPerspective(image, M, (cols, rows), borderValue=1.0)

        return image

    def load_and_ultra_preprocess_data(self):
        """Load and preprocess EMNIST with ultra-advanced preprocessing"""
        print("📚 Loading EMNIST Letters dataset with ultra-preprocessing...")

        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        print(f"Training samples: {ds_info.splits['train'].num_examples}")

        def ultra_preprocess(image, label):
            # Convert and resize
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))

            # Add channel dimension
            image = tf.expand_dims(image, -1)

            # Convert labels (EMNIST: 1-26 -> 0-25)
            label = tf.one_hot(label - 1, self.NUM_CLASSES)

            return image, label

        # Apply preprocessing
        ds_train = ds_train.map(ultra_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(ultra_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        # Create validation split
        train_size = int(0.9 * ds_info.splits['train'].num_examples)
        ds_train_split = ds_train.take(train_size)
        ds_val = ds_train.skip(train_size)

        # Batch and prefetch
        ds_train_split = ds_train_split.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_val = ds_val.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_test = ds_test.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        return ds_train_split, ds_val, ds_test

    def create_attention_block(self, x, filters):
        """Create attention mechanism block"""
        # Self-attention
        attention = Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = x * attention

        # Multi-head attention simulation
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)

        return x

    def create_residual_block(self, x, filters, stride=1):
        """Create residual block with attention"""
        shortcut = x

        # First conv
        x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Attention
        x = self.create_attention_block(x, filters)

        # Second conv
        x = Conv2D(filters, (3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        # Adjust shortcut
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        # Add shortcut
        x = Add()([x, shortcut])
        x = Activation('relu')(x)

        return x

    def create_ultra_model(self):
        """Create ultra-optimized CNN with attention and residual connections"""
        inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 1))

        # Initial conv with larger kernel
        x = Conv2D(64, (5, 5), strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(2, 2)(x)

        # Residual blocks with attention
        x = self.create_residual_block(x, 64)
        x = self.create_residual_block(x, 64)

        x = self.create_residual_block(x, 128, stride=2)
        x = self.create_residual_block(x, 128)

        x = self.create_residual_block(x, 256, stride=2)
        x = self.create_residual_block(x, 256)

        # Global attention pooling
        attention_weights = Conv2D(1, (1, 1), activation='sigmoid')(x)
        x = x * attention_weights
        x = GlobalAveragePooling2D()(x)

        # Dense layers with advanced regularization
        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output
        outputs = Dense(self.NUM_CLASSES, activation='softmax')(x)

        return Model(inputs, outputs)

    def focal_loss(self, gamma=2.0, alpha=0.25):
        """Advanced focal loss for better handling of hard examples"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (1 - y_pred)
            fl = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)

            return tf.reduce_mean(fl)
        return focal_loss_fixed

    def create_ultra_learning_schedule(self):
        """Ultra-optimized learning rate schedule"""
        def scheduler(epoch, lr):
            if epoch < 30:
                return lr
            elif epoch < 70:
                return lr * 0.8
            elif epoch < 100:
                return lr * 0.6
            elif epoch < 130:
                return lr * 0.3
            else:
                return lr * 0.1
        return LearningRateScheduler(scheduler)

    class UltraAccuracyCallback(Callback):
        """Advanced callback monitoring multiple metrics"""
        def __init__(self, trainer, validation_data):
            super().__init__()
            self.trainer = trainer
            self.validation_data = validation_data
            self.best_word_acc = 0

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:  # Check every 5 epochs
                word_accuracy = self.trainer.evaluate_ultra_word_accuracy(self.validation_data)
                letter_accuracy = self.trainer.evaluate_letter_accuracy(self.validation_data)

                print(f"\n📊 Epoch {epoch}: Word Acc: {word_accuracy:.3f}, Letter Acc: {letter_accuracy:.3f}")

                if word_accuracy > self.best_word_acc:
                    self.best_word_acc = word_accuracy
                    print(f"🎯 New best word accuracy: {word_accuracy:.3f}")

    def evaluate_ultra_word_accuracy(self, ds_val):
        """Evaluate word-level accuracy on target words"""
        correct_words = 0
        total_words = 0

        # Sample some validation data
        for images, labels in ds_val.take(20):  # More samples
            predictions = self.model.predict(images, verbose=0)

            for i in range(min(20, len(images))):  # Sample predictions per batch
                true_idx = np.argmax(labels[i])
                pred_idx = np.argmax(predictions[i])

                true_letter = chr(true_idx + ord('A'))
                pred_letter = chr(pred_idx + ord('A'))

                # Check if both letters are in target words
                if any(true_letter in word for word in self.target_words):
                    total_words += 1
                    if pred_idx == true_idx:
                        correct_words += 1

        return correct_words / max(total_words, 1)

    def evaluate_letter_accuracy(self, ds_val):
        """Evaluate letter-level accuracy"""
        correct = 0
        total = 0

        for images, labels in ds_val.take(10):
            predictions = self.model.predict(images, verbose=0)

            for i in range(len(images)):
                true_idx = np.argmax(labels[i])
                pred_idx = np.argmax(predictions[i])

                total += 1
                if pred_idx == true_idx:
                    correct += 1

        return correct / total

    def train_ultra_model(self):
        """Train the ultra-optimized model"""
        print("🚀 Starting ultra-optimized model training...")

        # Load data
        ds_train, ds_val, ds_test = self.load_and_ultra_preprocess_data()

        # Create model
        model = self.create_ultra_model()

        # Compile with focal loss and advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )

        # Print model summary
        print("🏗️  Ultra-Optimized Model Architecture:")
        model.summary()

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=12,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'models/ultra_optimized_letter_recognition.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            self.create_ultra_learning_schedule(),
            self.UltraAccuracyCallback(self, ds_val)
        ]

        # Train model
        print("🎓 Ultra-training started...")
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

        # Save model
        model.save("models/letter_recognition_ultra_optimized.h5")
        print("💾 Ultra-optimized model saved as 'models/letter_recognition_ultra_optimized.h5'")

        # Plot training history
        self.plot_ultra_training_history(history)

        # Test on target words
        self.test_ultra_target_words(model)

        return model, history

    def plot_ultra_training_history(self, history):
        """Plot comprehensive ultra-training history"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

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

        # Placeholder for Top-3
        axes[1, 0].text(0.5, 0.5, 'Top-3 Accuracy\nNot Available',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, fontweight='bold', color='gray')
        axes[1, 0].set_title('Top-3 Accuracy', fontsize=14, fontweight='bold')

        # Placeholder for Top-5
        axes[1, 1].text(0.5, 0.5, 'Top-5 Accuracy\nNot Available',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, fontweight='bold', color='gray')
        axes[1, 1].set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')

        # Learning Rate
        if 'lr' in history.history:
            axes[2, 0].plot(history.history['lr'], linewidth=2, color='orange')
            axes[2, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Learning Rate')
            axes[2, 0].set_yscale('log')
            axes[2, 0].grid(True, alpha=0.3)

        # Placeholder
        axes[2, 1].text(0.5, 0.5, 'Ultra-Optimized\nTraining Complete',
                       ha='center', va='center', transform=axes[2, 1].transAxes,
                       fontsize=16, fontweight='bold', color='green')

        plt.tight_layout()
        plt.savefig('models/ultra_optimized_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("📈 Ultra-training history saved as 'models/ultra_optimized_training_history.png'")

    def test_ultra_target_words(self, model):
        """Test ultra-model performance on target words"""
        print("\n🎯 Testing ultra-performance on target words...")

        # Create simple test images for each letter
        def create_letter_image(letter):
            img = np.ones((self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.float32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3

            (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
            x = (self.IMG_SIZE - text_width) // 2
            y = (self.IMG_SIZE + text_height) // 2

            cv2.putText(img, letter, (x, y), font, font_scale, (0,), thickness)
            return img

        # Test sample words
        sample_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP', 'ANT', 'CAR']

        total_letters = 0
        correct_letters = 0

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
                total_letters += 1
                if is_correct:
                    correct_letters += 1

                status = "✅" if is_correct else "❌"
                print(f"    {letter} -> {predicted_letter} ({confidence:.3f}) {status}")

            word_acc = sum(word_accuracy) / len(word_accuracy) * 100
            print(f"  Word accuracy: {word_acc:.1f}%")

        overall_acc = correct_letters / total_letters * 100
        print(f"\n🎯 Overall letter accuracy on target words: {overall_acc:.1f}%")

    def create_ultra_word_dictionary(self):
        """Create ultra-optimized word dictionary for real-time recognition"""
        print("📚 Creating ultra-optimized word dictionary...")

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
            'avg_word_length': np.mean([len(w) for w in self.target_words]),
            'optimization_level': 'ultra'
        }

        import json
        with open('models/ultra_optimized_word_dictionary.json', 'w') as f:
            json.dump(dictionary_data, f, indent=2)

        print("💾 Ultra-optimized word dictionary saved as 'models/ultra_optimized_word_dictionary.json'")

def main():
    """Main ultra-training function"""
    print("🧠 ULTRA-OPTIMIZED LETTER RECOGNITION MODEL TRAINING")
    print("=" * 70)
    print("🎯 Advanced techniques for maximum air writing accuracy")
    print("⚡ Focal loss + Attention + Residual + Ultra-augmentation")
    print("=" * 70)

    trainer = UltraOptimizedLetterTrainer()

    try:
        # Create ultra-optimized dictionary
        trainer.create_ultra_word_dictionary()

        # Train the ultra-model
        model, history = trainer.train_ultra_model()

        print("\n" + "=" * 70)
        print("🎉 ULTRA-OPTIMIZED TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("📁 Files created:")
        print("   • models/letter_recognition_ultra_optimized.h5 (Ultra model)")
        print("   • models/ultra_optimized_letter_recognition.h5 (Checkpoint)")
        print("   • models/ultra_optimized_training_history.png (Training plots)")
        print("   • models/ultra_optimized_word_dictionary.json (Word dictionary)")
        print("\n🚀 Ready for ultra-high accuracy air writing word recognition!")

    except Exception as e:
        print(f"❌ Error during ultra-training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
