"""
🧠 LSTM Text-to-Text Generation - Deep Learning NLP Project

This project demonstrates LSTM neural networks for text generation tasks.
Features: character-level generation, multiple architectures, training, and text generation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class LSTMTextGenerator:
    """LSTM-based text generation model"""
    
    def __init__(self, max_sequence_length=50, lstm_units=256, dropout_rate=0.3):
        self.max_sequence_length = max_sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
    def load_sample_texts(self):
        """Load sample texts for training"""
        print("📚 Loading sample texts...")
        
        sample_texts = [
            """To be, or not to be, that is the question:
            Whether 'tis nobler in the mind to suffer
            The slings and arrows of outrageous fortune,
            Or to take arms against a sea of troubles,
            And by opposing end them. To die—to sleep,
            No more; and by a sleep to say we end
            The heart-ache and the thousand natural shocks
            That flesh is heir to: 'tis a consummation
            Devoutly to be wish'd. To die, to sleep;
            To sleep, perchance to dream—ay, there's the rub:""",
            
            """The road not taken by Robert Frost
            Two roads diverged in a yellow wood,
            And sorry I could not travel both
            And be one traveler, long I stood
            And looked down one as far as I could
            To where it bent in the undergrowth;
            Then took the other, as just as fair,
            And having perhaps the better claim,
            Because it was grassy and wanted wear;
            Though as for that the passing there
            Had worn them really about the same."""
        ]
        
        print(f"✅ Loaded {len(sample_texts)} sample texts")
        return sample_texts
    
    def preprocess_text(self, texts):
        """Preprocess text data for LSTM training"""
        print("🔧 Preprocessing text data...")
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create character mappings
        unique_chars = sorted(set(combined_text))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(unique_chars)
        
        print(f"✅ Character vocabulary size: {self.vocab_size}")
        
        # Create sequences
        sequences = []
        next_chars = []
        
        for i in range(0, len(combined_text) - self.max_sequence_length):
            sequences.append(combined_text[i:i + self.max_sequence_length])
            next_chars.append(combined_text[i + self.max_sequence_length])
        
        # Convert to numerical format
        X = np.zeros((len(sequences), self.max_sequence_length, self.vocab_size), dtype=np.bool_)
        y = np.zeros((len(sequences), self.vocab_size), dtype=np.bool_)
        
        for i, sequence in enumerate(sequences):
            for t, char in enumerate(sequence):
                X[i, t, self.char_to_idx[char]] = 1
            y[i, self.char_to_idx[next_chars[i]]] = 1
        
        print(f"✅ Created {len(sequences)} training sequences")
        return X, y
    
    def build_model(self):
        """Build LSTM model architecture"""
        print("🏗️ Building LSTM model...")
        
        self.model = Sequential([
            LSTM(self.lstm_units, 
                 return_sequences=True, 
                 input_shape=(self.max_sequence_length, self.vocab_size)),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built and compiled successfully")
        self.model.summary()
        return self.model
    
    def train(self, X, y, epochs=50, batch_size=64, validation_split=0.2):
        """Train the LSTM model"""
        print("🚀 Starting model training...")
        
        # Callbacks
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        
        # Training
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed successfully")
        return history
    
    def generate_text(self, seed_text, length=100, temperature=1.0):
        """Generate text using the trained model"""
        if self.model is None:
            raise ValueError("Model must be trained before generating text")
        
        print(f"🎭 Generating text with temperature {temperature}...")
        
        generated_text = seed_text
        
        for _ in range(length):
            # Prepare input sequence
            sequence = np.zeros((1, self.max_sequence_length, self.vocab_size))
            for t, char in enumerate(generated_text[-self.max_sequence_length:]):
                if char in self.char_to_idx:
                    sequence[0, t, self.char_to_idx[char]] = 1
            
            # Predict next character
            pred = self.model.predict(sequence, verbose=0)[0]
            pred = np.log(pred) / temperature
            exp_pred = np.exp(pred)
            pred = exp_pred / np.sum(exp_pred)
            
            # Sample next character
            next_idx = np.random.choice(len(pred), p=pred)
            next_char = self.idx_to_char.get(next_idx, ' ')
            generated_text += next_char
        
        print("✅ Text generation completed")
        return generated_text
    
    def plot_training_history(self, history):
        """Plot training history"""
        print("📈 Creating training history plots...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('LSTM Training History', fontsize=16)
        
        # Loss plot
        ax1.plot(history.history['loss'], label='Training Loss', color='blue')
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Training history plots created")

def main():
    """Main function to run the LSTM text generation project"""
    print("🧠 LSTM Text-to-Text Generation - Deep Learning NLP Project")
    print("=" * 70)
    
    # Initialize text generator
    generator = LSTMTextGenerator(
        max_sequence_length=50,
        lstm_units=256,
        dropout_rate=0.3
    )
    
    # Load sample texts
    texts = generator.load_sample_texts()
    
    # Preprocess data
    print("\n🔧 Data Preprocessing:")
    X, y = generator.preprocess_text(texts)
    
    # Build model
    print("\n🏗️ Model Architecture:")
    model = generator.build_model()
    
    # Train model
    print("\n🚀 Model Training:")
    history = generator.train(X, y, epochs=50, batch_size=64)
    
    # Plot training history
    print("\n📈 Training Visualization:")
    generator.plot_training_history(history)
    
    # Generate sample texts
    print("\n🎭 Text Generation Examples:")
    
    seed_texts = [
        "To be or not to be",
        "The road not taken",
        "Machine learning is"
    ]
    
    for seed in seed_texts:
        print(f"\n🌱 Seed: '{seed}'")
        generated = generator.generate_text(seed, length=100, temperature=0.8)
        print(f"📝 Generated: {generated}")
    
    # Generate with different temperatures
    print("\n🌡️ Temperature Variations:")
    seed = "The future of AI"
    for temp in [0.5, 0.8, 1.0, 1.2]:
        generated = generator.generate_text(seed, length=50, temperature=temp)
        print(f"🌡️ T={temp}: {generated}")
    
    print(f"\n✅ LSTM Text Generation Project Completed Successfully!")
    print(f"📚 This project demonstrates:")
    print(f"   - LSTM architecture design and implementation")
    print(f"   - Text preprocessing and tokenization")
    print(f"   - Model training and optimization")
    print(f"   - Text generation with temperature sampling")

if __name__ == "__main__":
    main()
