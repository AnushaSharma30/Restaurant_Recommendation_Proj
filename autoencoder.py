import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE # Import SMOTE
import gc

def execute_stage_3_with_smote():
    """
    Loads processed data and uses SMOTE to handle extreme class imbalance
    before training the final model.
    """
    # ==============================================================================
    # STAGE 3 (WITH SMOTE): TRAINING ON BALANCED DATA
    # ==============================================================================
    print("\n--- STAGE 3 (WITH SMOTE): Training Final Predictive Model ---")
    
    try:
        final_modeling_df = pd.read_csv('data_with_embeddings.csv')
        print("Successfully loaded 'data_with_embeddings.csv'.")
    except FileNotFoundError:
        print("ERROR: 'data_with_embeddings.csv' not found. Please run the full script first.")
        return

    # --- 3a. Create the Target Variable ---
    target_variable = 'Hypertension'
    if target_variable not in final_modeling_df.columns:
        print(f"ERROR: '{target_variable}' column not found.")
        return
    print(f"Target Variable: Predicting '{target_variable}'")

    # --- 3b. Perform User-Level Train-Test Split ---
    print("Performing user-level train-test split...")
    all_user_ids = final_modeling_df['user_id'].unique()
    train_user_ids, test_user_ids = train_test_split(all_user_ids, test_size=0.2, random_state=42)
    train_df = final_modeling_df[final_modeling_df['user_id'].isin(train_user_ids)]
    test_df = final_modeling_df[final_modeling_df['user_id'].isin(test_user_ids)]

    # --- 3c. Prepare Features (X) and Target (y) ---
    features_to_use = [
        'age', 'user_lat', 'user_long', 'rest_lat', 'rest_long'
    ] + [col for col in final_modeling_df if 'taste_dna' in col]
    
    X_train = train_df[features_to_use]
    y_train = train_df[target_variable]
    
    X_test = test_df[features_to_use]
    y_test = test_df[target_variable]
    
    print(f"Original training data shape: {X_train.shape}")
    print(f"Class distribution in original training data:\n{y_train.value_counts(normalize=True)}")

    # --- 3d. Apply SMOTE to the Training Data ---
    print("\nApplying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    # IMPORTANT: Only apply SMOTE to the training data, never the test data.
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled training data shape: {X_train_resampled.shape}")
    print(f"Class distribution in resampled training data:\n{y_train_resampled.value_counts(normalize=True)}")
    
    # --- 3e. Train and Evaluate the Final Model ---
    print("\nTraining Random Forest Classifier on the balanced data...")
    # We no longer need class_weight='balanced' because the data itself is balanced.
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(X_train_resampled, y_train_resampled)

    print("Evaluating model performance on the original, unbalanced test set...")
    predictions = final_model.predict(X_test)
    
    print("\n--- (FINAL) Model Evaluation Report with SMOTE ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("--------------------------------------------------")

if __name__ == '__main__':
    execute_stage_3_with_smote()