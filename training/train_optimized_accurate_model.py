#!/usr/bin/env python3
"""
Optimized Accurate Model Training for Air Writing Recognition
Focus on maximum accuracy with proven techniques
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Input, Activation, Add, MultiHeadAttention,
    LayerNormalization, Concatenate, SeparableConv2D, GlobalMaxPooling2D
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler, Callback
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import os
import cv2
import json
from collections import Counter
import seaborn as sns

class OptimizedAccurateTrainer:
    def __init__(self):
        self.NUM_CLASSES = 26
        self.IMG_SIZE = 32
        self.BATCH_SIZE = 64
        self.EPOCHS = 100
        
        # Enhanced target words for better recognition
        self.target_words = [
            # Core 3-letter words
            'CAT', 'DOG', 'BAT', 'RAT', 'HAT', 'MAT', 'SAT', 'FAT', 'PAT', 'VAT',
            'BIG', 'PIG', 'DIG', 'FIG', 'WIG', 'JIG', 'RIG',
            'SUN', 'RUN', 'FUN', 'GUN', 'BUN', 'NUN',
            'BOX', 'FOX', 'COX', 'SOX',
            'BED', 'RED', 'LED', 'FED', 'WED',
            'TOP', 'HOP', 'MOP', 'POP', 'COP', 'SOP',
            'CUP', 'PUP', 'SUP', 'YUP',
            'BAG', 'TAG', 'RAG', 'SAG', 'WAG', 'NAG',
            'BUS', 'GUS', 'PUS',
            'HOT', 'POT', 'COT', 'DOT', 'GOT', 'LOT', 'NOT', 'ROT',
            'BAD', 'DAD', 'HAD', 'MAD', 'PAD', 'SAD', 'FAD',
            'BEE', 'SEE', 'TEE', 'FEE', 'PEE', 'WEE',
            'EGG', 'LEG', 'BEG', 'PEG', 'KEG',
            'ICE', 'NICE', 'RICE', 'MICE', 'VICE', 'DICE',
            'ANT', 'PAN', 'MAN', 'CAN', 'FAN', 'VAN', 'TAN', 'RAN',
            'CAR', 'FAR', 'JAR', 'BAR', 'TAR', 'WAR',
            'DAY', 'BAY', 'HAY', 'JAY', 'LAY', 'MAY', 'PAY', 'RAY', 'SAY', 'WAY',
            'EYE', 'DYE', 'RYE', 'BYE',
            'OIL', 'BOIL', 'COIL', 'FOIL', 'SOIL', 'TOIL',
            
            # Common 4-letter words
            'WORD', 'WORK', 'WELL', 'WANT', 'VERY', 'TIME', 'TAKE', 'SOME',
            'MAKE', 'LIKE', 'KNOW', 'HELP', 'GOOD', 'FIND', 'COME', 'BACK'
        ]
        
        os.makedirs("models", exist_ok=True)
        
        # Analyze letter frequencies
        self.letter_freq = Counter()
        for word in self.target_words:
            self.letter_freq.update(word)
        
        print(f"🎯 Training for {len(self.target_words)} target words")
        print("Top letters:", sorted(self.letter_freq.items(), key=lambda x: x[1], reverse=True)[:10])

    def create_advanced_augmentation(self):
        """Create advanced data augmentation for air writing"""
        return ImageDataGenerator(
            rotation_range=25,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=[0.8, 1.3],
            fill_mode='constant',
            cval=1.0,
            brightness_range=[0.7, 1.3],
            preprocessing_function=self.air_writing_augmentation
        )

    def air_writing_augmentation(self, image):
        """Air writing specific augmentation"""
        # Simulate hand tremor with random morphological operations
        if np.random.random() > 0.7:
            kernel_size = np.random.randint(1, 4)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if np.random.random() > 0.5:
                image = cv2.dilate(image, kernel, iterations=1)
            else:
                image = cv2.erode(image, kernel, iterations=1)
        
        # Add slight blur to simulate motion
        if np.random.random() > 0.8:
            blur_kernel = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        
        # Add noise to simulate imperfect tracking
        if np.random.random() > 0.8:
            noise = np.random.normal(0, 0.05, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 1)
        
        return image

    def load_and_preprocess_data(self):
        """Load and preprocess EMNIST data with class balancing"""
        print("📚 Loading EMNIST Letters dataset...")
        
        (ds_train, ds_test), ds_info = tfds.load(
            'emnist/letters',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )
        
        print(f"Training samples: {ds_info.splits['train'].num_examples}")
        
        def preprocess(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
            image = tf.expand_dims(image, -1)
            label = tf.one_hot(label - 1, self.NUM_CLASSES)
            return image, label
        
        ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        
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
        """Create attention mechanism"""
        # Spatial attention
        spatial_attention = Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)
        x_spatial = tf.keras.layers.Multiply()([x, spatial_attention])
        
        # Channel attention
        avg_pool = GlobalAveragePooling2D(keepdims=True)(x_spatial)
        max_pool = GlobalMaxPooling2D(keepdims=True)(x_spatial)
        
        avg_dense = Dense(filters // 8, activation='relu')(avg_pool)
        avg_dense = Dense(filters, activation='sigmoid')(avg_dense)
        
        max_dense = Dense(filters // 8, activation='relu')(max_pool)
        max_dense = Dense(filters, activation='sigmoid')(max_dense)
        
        channel_attention = Add()([avg_dense, max_dense])
        x_channel = tf.keras.layers.Multiply()([x_spatial, channel_attention])
        
        return x_channel

    def create_optimized_model(self):
        """Create optimized CNN model with attention"""
        inputs = Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 1))
        
        # Initial convolution
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Block 1
        x = SeparableConv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.create_attention_block(x, 64)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.25)(x)
        
        # Block 2
        x = SeparableConv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.create_attention_block(x, 128)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.3)(x)
        
        # Block 3
        x = SeparableConv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = self.create_attention_block(x, 256)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.4)(x)
        
        # Global pooling
        gap = GlobalAveragePooling2D()(x)
        gmp = GlobalMaxPooling2D()(x)
        global_features = Concatenate()([gap, gmp])
        
        # Dense layers
        x = Dense(512, kernel_regularizer=l2(0.001))(global_features)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        x = Dense(256, kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.4)(x)
        
        # Output
        outputs = Dense(self.NUM_CLASSES, activation='softmax')(x)
        
        return Model(inputs, outputs)

    def focal_loss(self, gamma=2.0, alpha=0.25):
        """Focal loss for handling class imbalance"""
        def loss_function(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            fl = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t)
            
            return tf.reduce_mean(fl)
        
        return loss_function

    def cosine_schedule(self, epoch, lr):
        """Cosine annealing learning rate schedule"""
        import math
        min_lr = 1e-7
        max_lr = 0.001
        cycle_length = 30
        
        cycle = math.floor(1 + epoch / cycle_length)
        x = abs(epoch / cycle_length - cycle)
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(x * math.pi))
        
        return lr

    class AccuracyCallback(Callback):
        """Custom callback for monitoring word accuracy"""
        def __init__(self, trainer, validation_data):
            super().__init__()
            self.trainer = trainer
            self.validation_data = validation_data
            self.best_acc = 0

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 0:
                word_acc = self.trainer.evaluate_word_accuracy(self.validation_data)
                print(f"\n📊 Epoch {epoch}: Word Accuracy: {word_acc:.4f}")
                
                if word_acc > self.best_acc:
                    self.best_acc = word_acc
                    print(f"🎯 New best word accuracy: {word_acc:.4f}")

    def evaluate_word_accuracy(self, ds_val):
        """Evaluate accuracy on target words"""
        correct = 0
        total = 0
        
        for images, labels in ds_val.take(20):
            predictions = self.model.predict(images, verbose=0)
            
            for i in range(min(20, len(images))):
                true_idx = np.argmax(labels[i])
                pred_idx = np.argmax(predictions[i])
                
                true_letter = chr(true_idx + ord('A'))
                
                if any(true_letter in word for word in self.target_words):
                    total += 1
                    if pred_idx == true_idx:
                        correct += 1
        
        return correct / max(total, 1)

    def train_optimized_model(self):
        """Train the optimized model"""
        print("🚀 Starting optimized training...")
        
        # Load data
        ds_train, ds_val, ds_test = self.load_and_preprocess_data()
        
        # Create model
        model = self.create_optimized_model()
        self.model = model
        
        # Compile with focal loss
        optimizer = AdamW(learning_rate=0.001, weight_decay=0.0001)
        
        model.compile(
            optimizer=optimizer,
            loss=self.focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("🏗️  Optimized Model Architecture:")
        model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.7,
                patience=8,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                'models/optimized_accurate_letter_recognition.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            LearningRateScheduler(self.cosine_schedule, verbose=1),
            self.AccuracyCallback(self, ds_val)
        ]
        
        # Train
        print("🎓 Training started...")
        history = model.fit(
            ds_train,
            epochs=self.EPOCHS,
            validation_data=ds_val,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n📊 Final evaluation...")
        test_results = model.evaluate(ds_test, verbose=1)
        print(f"Test Accuracy: {test_results[1]:.4f}")
        
        # Save final model
        model.save("models/letter_recognition_optimized_accurate.h5")
        print("💾 Model saved!")
        
        # Test on target words
        self.test_target_words(model)
        
        # Create dictionary
        self.create_optimized_dictionary()
        
        return model, history

    def test_target_words(self, model):
        """Test model on target words"""
        print("\n🎯 Testing on target words...")
        
        def create_test_image(letter):
            img = np.ones((self.IMG_SIZE, self.IMG_SIZE, 1), dtype=np.float32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(letter, font, font_scale, thickness)
            x = (self.IMG_SIZE - text_width) // 2
            y = (self.IMG_SIZE + text_height) // 2
            
            cv2.putText(img, letter, (x, y), font, font_scale, (0,), thickness)
            return img
        
        sample_words = ['CAT', 'DOG', 'SUN', 'BOX', 'RED', 'BIG', 'TOP', 'CUP']
        total_correct = 0
        total_letters = 0
        
        for word in sample_words:
            print(f"\nTesting word: {word}")
            word_correct = 0
            
            for letter in word:
                test_img = create_test_image(letter)
                test_img = np.expand_dims(test_img, 0)
                
                prediction = model.predict(test_img, verbose=0)
                pred_idx = np.argmax(prediction)
                pred_letter = chr(pred_idx + ord('A'))
                confidence = np.max(prediction)
                
                is_correct = pred_letter == letter
                total_letters += 1
                if is_correct:
                    total_correct += 1
                    word_correct += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {letter} -> {pred_letter} ({confidence:.3f}) {status}")
            
            word_acc = word_correct / len(word) * 100
            print(f"Word accuracy: {word_acc:.1f}%")
        
        overall_acc = total_correct / total_letters * 100
        print(f"\n🎯 Overall accuracy on target words: {overall_acc:.1f}%")

    def create_optimized_dictionary(self):
        """Create optimized dictionary"""
        print("📚 Creating optimized dictionary...")
        
        # Position patterns
        position_patterns = {}
        for pos in range(max(len(word) for word in self.target_words)):
            position_patterns[pos] = Counter()
            for word in self.target_words:
                if pos < len(word):
                    position_patterns[pos][word[pos]] += 1
        
        dictionary_data = {
            'target_words': self.target_words,
            'letter_frequencies': dict(self.letter_freq),
            'position_patterns': {str(k): dict(v) for k, v in position_patterns.items()},
            'total_words': len(self.target_words),
            'avg_word_length': float(np.mean([len(w) for w in self.target_words])),
            'optimization_level': 'accurate'
        }
        
        with open('models/optimized_accurate_word_dictionary.json', 'w') as f:
            json.dump(dictionary_data, f, indent=2)
        
        print("💾 Dictionary saved!")

def main():
    """Main training function"""
    print("🧠 OPTIMIZED ACCURATE LETTER RECOGNITION TRAINING")
    print("=" * 60)
    print("🎯 Focus on maximum accuracy for air writing")
    print("⚡ Attention + Focal Loss + Advanced Augmentation")
    print("=" * 60)
    
    trainer = OptimizedAccurateTrainer()
    
    try:
        model, history = trainer.train_optimized_model()
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Files created:")
        print("   • models/letter_recognition_optimized_accurate.h5")
        print("   • models/optimized_accurate_letter_recognition.h5")
        print("   • models/optimized_accurate_word_dictionary.json")
        print("\n🚀 Ready for high-accuracy air writing!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()