import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
import gc
import os

def execute_pathway_b_chunked():
    """
    Executes the deep learning pathway by processing the main CSV in chunks
    to avoid memory errors.
    """
    # ==============================================================================
    # STAGE 1: LEARNING THE "TASTE DNA" WITH AN AUTOENCODER (CHUNK BY CHUNK)
    # ==============================================================================
    print("--- STAGE 1: Learning User Taste DNA (Chunk by Chunk) ---")

    INPUT_FILENAME = 'flattened_hybrid_craving_final.csv'
    CHUNKSIZE = 10000  # Read 10,000 rows at a time
    
    try:
        # Check if the file exists before starting
        if not os.path.exists(INPUT_FILENAME):
            print(f"ERROR: '{INPUT_FILENAME}' not found. Please ensure it is in the same directory.")
            return
    except Exception as e:
        print(f"An error occurred checking for the file: {e}")
        return

    # --- 1a. Initialize the Autoencoder Model ---
    # We need to know the dimensions, so we'll read just one row to get the column names.
    temp_df = pd.read_csv(INPUT_FILENAME, nrows=1)
    behavioral_cols = [col for col in temp_df.columns if 'menu_item' in col or 'craving' in col]
    
    input_dim = len(behavioral_cols)
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128)(input_layer)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Dense(64)(encoder)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Dense(encoding_dim, activation='tanh')(encoder)
    decoder = Dense(64)(encoder)
    decoder = LeakyReLU(alpha=0.2)(decoder)
    decoder = Dense(128)(decoder)
    decoder = LeakyReLU(alpha=0.2)(decoder)
    decoder = Dense(input_dim, activation=None)(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    print("Autoencoder model built successfully.")
    
    # --- 1b. Train the Autoencoder by iterating through file chunks ---
    print("Training Autoencoder on file chunks...")
    scaler = StandardScaler()
    epochs = 5 # Fewer epochs since we are iterating over the data multiple times

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        chunk_iterator = pd.read_csv(INPUT_FILENAME, chunksize=CHUNKSIZE)
        for i, chunk in enumerate(chunk_iterator):
            print(f"  - Training on chunk {i+1}...")
            # Extract and scale behavioral data for the current chunk
            behavioral_chunk = chunk[behavioral_cols]
            # Use fit_transform on the first chunk, then just transform
            if i == 0 and epoch == 0:
                behavioral_chunk_scaled = scaler.fit_transform(behavioral_chunk)
            else:
                behavioral_chunk_scaled = scaler.transform(behavioral_chunk)
            
            # Train the model on this single batch of data
            autoencoder.train_on_batch(behavioral_chunk_scaled, behavioral_chunk_scaled)
        gc.collect()

    print("Autoencoder training complete.")
    
    # ==============================================================================
    # STAGE 2: GENERATING EMBEDDINGS AND SAVING TO A NEW FILE
    # ==============================================================================
    print("\n--- STAGE 2: Generating Embeddings and Creating New Dataset ---")
    
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    PROCESSED_FILENAME = 'data_with_embeddings.csv'
    
    # Process the file in chunks again, this time to create the new dataset
    is_first_chunk = True
    chunk_iterator = pd.read_csv(INPUT_FILENAME, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(chunk_iterator):
        print(f"  - Processing chunk {i+1} to generate embeddings...")
        
        # Get embeddings for the chunk
        behavioral_chunk = chunk[behavioral_cols]
        behavioral_chunk_scaled = scaler.transform(behavioral_chunk)
        taste_dna_embeddings = encoder_model.predict(behavioral_chunk_scaled)
        
        # Create the new DataFrame
        embedding_cols = [f'taste_dna_{i}' for i in range(encoding_dim)]
        embeddings_df = pd.DataFrame(taste_dna_embeddings, columns=embedding_cols, index=chunk.index)
        
        non_behavioral_cols = [col for col in chunk.columns if col not in behavioral_cols]
        processed_chunk = pd.concat([chunk[non_behavioral_cols], embeddings_df], axis=1)
        
        # Save chunk to a new file
        if is_first_chunk:
            processed_chunk.to_csv(PROCESSED_FILENAME, index=False, mode='w', header=True)
            is_first_chunk = False
        else:
            processed_chunk.to_csv(PROCESSED_FILENAME, index=False, mode='a', header=False)
        
        gc.collect()

    print(f"New dataset saved as '{PROCESSED_FILENAME}'.")

    # ==============================================================================
    # STAGE 3: TRAINING THE FINAL PREDICTIVE MODEL
    # ==============================================================================
    print("\n--- STAGE 3: Training Final Predictive Model ---")
    
    # Now, load the new, smaller dataset (should fit in memory)
    final_modeling_df = pd.read_csv(PROCESSED_FILENAME)

    # --- 3a. Create Target Variable ---
    menu_item_cols = [col for col in final_modeling_df.columns if 'menu_item' in col]
    final_modeling_df['total_orders'] = final_modeling_df[menu_item_cols].sum(axis=1)
    median_orders = final_modeling_df['total_orders'].median()
    final_modeling_df['is_high_frequency'] = (final_modeling_df['total_orders'] > median_orders).astype(int)
    print(f"Created target variable 'is_high_frequency'. Median orders: {median_orders}")

    # --- 3b. User-Level Split ---
    print("Performing user-level train-test split...")
    all_user_ids = final_modeling_df['user_id'].unique()
    train_user_ids, test_user_ids = train_test_split(all_user_ids, test_size=0.2, random_state=42)
    train_df = final_modeling_df[final_modeling_df['user_id'].isin(train_user_ids)]
    test_df = final_modeling_df[final_modeling_df['user_id'].isin(test_user_ids)]

    # --- 3c. Prepare Features and Target ---
    features_to_use = ['age', 'user_lat', 'user_long', 'rest_lat', 'rest_long'] + [col for col in final_modeling_df if 'taste_dna' in col]
    X_train, y_train = train_df[features_to_use], train_df['is_high_frequency']
    X_test, y_test = test_df[features_to_use], test_df['is_high_frequency']
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    # --- 3d. Train and Evaluate ---
    print("Training Random Forest Classifier...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)
    
    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))
    print("-----------------------------")

if __name__ == '__main__':
    execute_pathway_b_chunked()