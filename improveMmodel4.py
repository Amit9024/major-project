import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class NanophotonicModelImproved:
    def __init__(self, data_path, separate_models=True, physics_informed=True):
        """
        Initialize the improved nanophotonic model
        
        Args:
            data_path: Path to the Excel file with the data
            separate_models: Whether to train separate models for each refractive index
            physics_informed: Whether to use physics-informed features
        """
        self.data_path = data_path
        self.separate_models = separate_models
        self.physics_informed = physics_informed
        self.models = {}  # Dictionary to store models for each refractive index
        self.scalers = {}  # Dictionary to store scalers
        
        # Available refractive indices in the dataset
        self.refractive_indices = [1.0, 1.2, 1.3, 1.33, 1.4, 1.5]
        
        # Directory to save models
        self.model_dir = "improved_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_data(self):
        """
        Load data from Excel file with multiple sheets
        
        Returns:
            DataFrame with all data combined
        """
        # Mapping of sheet names to refractive indices
        sheet_to_ri = {
            '1': 1.0,
            '2': 1.2,
            '3': 1.3,
            '4': 1.33,
            '5': 1.4,
            '6': 1.5
        }
        
        all_data = []
        
        for sheet_name, ri in sheet_to_ri.items():
            try:
                # Read the sheet
                df = pd.read_excel(self.data_path, sheet_name=sheet_name)
                
                # Normalize column names
                df.columns = [col.strip().lower() for col in df.columns]
                
                # Map to consistent column names
                column_mapping = {
                    'size (diameter)': 'diameter',
                    'refractive index of surrounding': 'refractive_index',
                    'resonance wavelength': 'resonance_wavelength',
                    'resonance intensity (extinction)': 'extinction'
                }
                
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Add refractive index if not already present
                if 'refractive_index' not in df.columns:
                    df['refractive_index'] = ri
                
                all_data.append(df)
                print(f"Successfully loaded sheet {sheet_name} with refractive index {ri}")
                
            except Exception as e:
                print(f"Error loading sheet {sheet_name}: {e}")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Print column names to verify
        print("Available columns:", combined_data.columns.tolist())
        print(f"Loaded {len(combined_data)} data points")
        
        # Check for NaN or infinite values
        if combined_data.isna().any().any() or np.isinf(combined_data.values).any():
            print("Warning: Dataset contains NaN or infinite values. Cleaning...")
            combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"After cleaning: {len(combined_data)} data points")
        
        return combined_data
    
    def add_physics_features(self, df):
        """
        Add physics-based features that might improve model performance
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Size parameter (2πr/λ = πd/λ) - related to Mie theory
        # This is approximated using the resonance wavelength
        df['size_parameter'] = np.pi * df['diameter'] / df['resonance_wavelength']
        
        # Squared and cubed terms to capture non-linear relationships
        df['diameter_squared'] = df['diameter'] ** 2
        
        # Interaction term (diameter * refractive_index)
        df['d_times_ri'] = df['diameter'] * df['refractive_index']
        
        # Optical density approximation
        df['optical_density'] = df['refractive_index'] * df['diameter'] / 100
        
        return df
    
    def analyze_data(self, df):
        """
        Analyze data to identify potential issues and patterns
        
        Args:
            df: DataFrame with the data
        """
        # Plot resonance wavelength vs diameter for each refractive index
        plt.figure(figsize=(12, 8))
        for ri in self.refractive_indices:
            subset = df[df['refractive_index'] == ri]
            plt.scatter(subset['diameter'], subset['resonance_wavelength'], 
                      alpha=0.5, label=f'RI = {ri}')
        
        plt.title('Resonance Wavelength vs Diameter by Refractive Index')
        plt.xlabel('Diameter (nm)')
        plt.ylabel('Resonance Wavelength (nm)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, 'wavelength_analysis.png'))
        
        # Plot extinction vs diameter for each refractive index
        plt.figure(figsize=(12, 8))
        for ri in self.refractive_indices:
            subset = df[df['refractive_index'] == ri]
            plt.scatter(subset['diameter'], subset['extinction'], 
                      alpha=0.5, label=f'RI = {ri}')
        
        plt.title('Extinction vs Diameter by Refractive Index')
        plt.xlabel('Diameter (nm)')
        plt.ylabel('Extinction')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.model_dir, 'extinction_analysis.png'))
        
        # Look for outliers using z-score
        for ri in self.refractive_indices:
            subset = df[df['refractive_index'] == ri]
            
            # Calculate z-scores for wavelength and extinction
            z_wavelength = (subset['resonance_wavelength'] - subset['resonance_wavelength'].mean()) / subset['resonance_wavelength'].std()
            z_extinction = (subset['extinction'] - subset['extinction'].mean()) / subset['extinction'].std()
            
            # Find potential outliers (|z| > 3)
            wavelength_outliers = subset[abs(z_wavelength) > 3]
            extinction_outliers = subset[abs(z_extinction) > 3]
            
            if len(wavelength_outliers) > 0:
                print(f"Found {len(wavelength_outliers)} potential outliers in wavelength for RI = {ri}")
            
            if len(extinction_outliers) > 0:
                print(f"Found {len(extinction_outliers)} potential outliers in extinction for RI = {ri}")
    
    def build_model(self, input_dim):
        """
        Build an improved model architecture
        
        Args:
            input_dim: Dimension of input features
            
        Returns:
            Compiled Keras model
        """
        inputs = Input(shape=(input_dim,))
        
        # First branch - simpler path
        x1 = Dense(32, activation='relu')(inputs)
        x1 = BatchNormalization()(x1)
        
        # Second branch - deeper path
        x2 = Dense(64, activation='relu')(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Dense(128, activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        x2 = Dense(64, activation='relu')(x2)
        x2 = BatchNormalization()(x2)
        
        # Combine branches
        combined = Concatenate()([x1, x2])
        combined = Dense(64, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.2)(combined)
        
        # Output layer (no activation for regression)
        outputs = Dense(2)(combined)
        
        # Create and compile model
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_data_for_training(self, df, ri=None):
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with all data
            ri: If provided, filter data for this refractive index
            
        Returns:
            X_train, X_test, y_train, y_test, X_scaler, y_scaler
        """
        # Filter by refractive index if specified
        if ri is not None:
            df = df[df['refractive_index'] == ri].copy()
        else:
            df = df.copy()
        
        # Add physics-based features if enabled
        if self.physics_informed:
            df = self.add_physics_features(df)
        
        # Select features and targets
        if self.physics_informed:
            X = df[['diameter', 'refractive_index', 'diameter_squared', 
                   'd_times_ri', 'optical_density']].values
        else:
            X = df[['diameter', 'refractive_index']].values
        
        y = df[['resonance_wavelength', 'extinction']].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        
        # Scale targets
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler, X_train, y_train, X_test, y_test
    
    def train_model(self, combined_df):
        """
        Train the model(s)
        
        Args:
            combined_df: DataFrame with all combined data
            
        Returns:
            Dictionary of trained models
        """
        if self.separate_models:
            # Train separate model for each refractive index
            for ri in self.refractive_indices:
                print(f"\nTraining model for refractive index {ri}")
                
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler, X_train, y_train, X_test, y_test = self.prepare_data_for_training(combined_df, ri)
                
                # Get input dimension
                input_dim = X_train_scaled.shape[1]
                
                # Build model
                model = self.build_model(input_dim)
                
                # Setup callbacks
                callbacks = [
                    EarlyStopping(patience=20, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5),
                    ModelCheckpoint(
                        filepath=os.path.join(self.model_dir, f'model_ri_{ri:.2f}.h5'),
                        save_best_only=True
                    )
                ]
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train_scaled,
                    epochs=300,
                    batch_size=32,
                    validation_data=(X_test_scaled, y_test_scaled),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Plot training history
                self.plot_training_history(history, ri)
                
                # Evaluate model
                y_pred_scaled = model.predict(X_test_scaled)
                y_pred = y_scaler.inverse_transform(y_pred_scaled)
                
                # Calculate metrics
                rmse_wavelength = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
                rmse_extinction = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
                r2_wavelength = r2_score(y_test[:, 0], y_pred[:, 0])
                r2_extinction = r2_score(y_test[:, 1], y_pred[:, 1])
                
                print(f"RI = {ri} - RMSE Wavelength: {rmse_wavelength:.4f}, R² Wavelength: {r2_wavelength:.4f}")
                print(f"RI = {ri} - RMSE Extinction: {rmse_extinction:.6f}, R² Extinction: {r2_extinction:.4f}")
                
                # Plot predictions vs actual
                self.plot_predictions(y_test, y_pred, ri)
                
                # Store model and scalers
                self.models[ri] = model
                self.scalers[ri] = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
                
                # Save scalers
                np.save(os.path.join(self.model_dir, f'X_scaler_ri_{ri:.2f}.npy'), 
                       [X_scaler.mean_, X_scaler.scale_])
                np.save(os.path.join(self.model_dir, f'y_scaler_ri_{ri:.2f}.npy'), 
                       [y_scaler.mean_, y_scaler.scale_])
        else:
            # Train a single model for all refractive indices
            print("\nTraining unified model for all refractive indices")
            
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler, X_train, y_train, X_test, y_test = self.prepare_data_for_training(combined_df)
            
            # Get input dimension
            input_dim = X_train_scaled.shape[1]
            
            # Build model
            model = self.build_model(input_dim)
            
            # Setup callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'unified_model.h5'),
                    save_best_only=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train_scaled,
                epochs=300,
                batch_size=32,
                validation_data=(X_test_scaled, y_test_scaled),
                callbacks=callbacks,
                verbose=1
            )
            
            # Plot training history
            self.plot_training_history(history)
            
            # Evaluate model
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            
            # Calculate metrics
            rmse_wavelength = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
            rmse_extinction = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
            r2_wavelength = r2_score(y_test[:, 0], y_pred[:, 0])
            r2_extinction = r2_score(y_test[:, 1], y_pred[:, 1])
            
            print(f"Unified Model - RMSE Wavelength: {rmse_wavelength:.4f}, R² Wavelength: {r2_wavelength:.4f}")
            print(f"Unified Model - RMSE Extinction: {rmse_extinction:.6f}, R² Extinction: {r2_extinction:.4f}")
            
            # Plot predictions vs actual
            self.plot_predictions(y_test, y_pred)
            
            # Store model and scalers
            self.models['unified'] = model
            self.scalers['unified'] = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
            
            # Save scalers
            np.save(os.path.join(self.model_dir, 'X_scaler_unified.npy'), 
                   [X_scaler.mean_, X_scaler.scale_])
            np.save(os.path.join(self.model_dir, 'y_scaler_unified.npy'), 
                   [y_scaler.mean_, y_scaler.scale_])
            
        return self.models
    
    def plot_training_history(self, history, ri=None):
        """
        Plot the training history
        
        Args:
            history: Training history object
            ri: Refractive index (optional)
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        if ri is not None:
            plt.title(f'Model Loss (RI = {ri})')
        else:
            plt.title('Model Loss (Unified)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        if ri is not None:
            plt.title(f'Model MAE (RI = {ri})')
        else:
            plt.title('Model MAE (Unified)')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if ri is not None:
            plt.savefig(os.path.join(self.model_dir, f'training_history_ri_{ri:.2f}.png'))
        else:
            plt.savefig(os.path.join(self.model_dir, 'training_history_unified.png'))
        
        plt.close()
    
    def plot_predictions(self, y_test, y_pred, ri=None):
        """
        Plot predicted vs actual values
        
        Args:
            y_test: Actual test values
            y_pred: Predicted values
            ri: Refractive index (optional)
        """
        plt.figure(figsize=(14, 6))
        
        # Plot resonance wavelength predictions
        plt.subplot(1, 2, 1)
        plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
        plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                [y_test[:, 0].min(), y_test[:, 0].max()], 
                'r--')
        if ri is not None:
            plt.title(f'Predicted vs Actual Resonance Wavelength (RI = {ri})')
        else:
            plt.title('Predicted vs Actual Resonance Wavelength (Unified)')
        plt.xlabel('Actual Wavelength (nm)')
        plt.ylabel('Predicted Wavelength (nm)')
        plt.grid(True)
        
        # Plot extinction predictions
        plt.subplot(1, 2, 2)
        plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
        plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
                [y_test[:, 1].min(), y_test[:, 1].max()], 
                'r--')
        if ri is not None:
            plt.title(f'Predicted vs Actual Extinction (RI = {ri})')
        else:
            plt.title('Predicted vs Actual Extinction (Unified)')
        plt.xlabel('Actual Extinction')
        plt.ylabel('Predicted Extinction')
        plt.grid(True)
        
        plt.tight_layout()
        
        if ri is not None:
            plt.savefig(os.path.join(self.model_dir, f'predictions_ri_{ri:.2f}.png'))
        else:
            plt.savefig(os.path.join(self.model_dir, 'predictions_unified.png'))
        
        plt.close()
    
    def predict(self, diameter, refractive_index):
        """
        Make a prediction using the trained model
        
        Args:
            diameter: Particle diameter (nm)
            refractive_index: Refractive index of surrounding medium
            
        Returns:
            Predicted resonance wavelength and extinction
        """
        if self.separate_models:
            # Find closest available refractive index
            closest_ri = min(self.refractive_indices, key=lambda x: abs(x - refractive_index))
            if abs(closest_ri - refractive_index) > 0.01:
                print(f"Warning: Using closest available RI: {closest_ri}")
            
            # Get model and scalers for this RI
            model = self.models[closest_ri]
            X_scaler = self.scalers[closest_ri]['X_scaler']
            y_scaler = self.scalers[closest_ri]['y_scaler']
            
            # Prepare input features
            if self.physics_informed:
                # For physics-informed features, we need an initial guess for the wavelength
                # We'll use a simple approximation based on the diameter
                approx_wavelength = 300 + 1.5 * diameter
                
                # Create the feature vector with physics-informed features
                X_pred = np.array([[
                    diameter, 
                    closest_ri,
                    diameter ** 2,
                    diameter * closest_ri,
                    closest_ri * diameter / 100
                ]])
            else:
                X_pred = np.array([[diameter, closest_ri]])
        else:
            # Use unified model
            model = self.models['unified']
            X_scaler = self.scalers['unified']['X_scaler']
            y_scaler = self.scalers['unified']['y_scaler']
            
            # Prepare input features
            if self.physics_informed:
                # For physics-informed features, we need an initial guess for the wavelength
                # We'll use a simple approximation based on the diameter
                approx_wavelength = 300 + 1.5 * diameter
                
                # Create the feature vector with physics-informed features
                X_pred = np.array([[
                    diameter, 
                    refractive_index,
                    diameter ** 2,
                    diameter * refractive_index,
                    refractive_index * diameter / 100
                ]])
            else:
                X_pred = np.array([[diameter, refractive_index]])
        
        # Scale input
        X_pred_scaled = X_scaler.transform(X_pred)
        
        # Make prediction
        y_pred_scaled = model.predict(X_pred_scaled)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        return y_pred[0]  # [resonance_wavelength, extinction]
    
    def load_saved_models(self):
        """
        Load saved models from disk
        """
        if self.separate_models:
            for ri in self.refractive_indices:
                model_path = os.path.join(self.model_dir, f'model_ri_{ri:.2f}.h5')
                if os.path.exists(model_path):
                    self.models[ri] = tf.keras.models.load_model(model_path)
                    
                    # Load scalers
                    X_scaler = StandardScaler()
                    X_scaler_params = np.load(os.path.join(self.model_dir, f'X_scaler_ri_{ri:.2f}.npy'), 
                                            allow_pickle=True)
                    X_scaler.mean_ = X_scaler_params[0]
                    X_scaler.scale_ = X_scaler_params[1]
                    
                    y_scaler = StandardScaler()
                    y_scaler_params = np.load(os.path.join(self.model_dir, f'y_scaler_ri_{ri:.2f}.npy'), 
                                            allow_pickle=True)
                    y_scaler.mean_ = y_scaler_params[0]
                    y_scaler.scale_ = y_scaler_params[1]
                    
                    self.scalers[ri] = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
                    
                    print(f"Loaded model for RI = {ri}")
        else:
            model_path = os.path.join(self.model_dir, 'unified_model.h5')
            if os.path.exists(model_path):
                self.models['unified'] = tf.keras.models.load_model(model_path)
                
                # Load scalers
                X_scaler = StandardScaler()
                X_scaler_params = np.load(os.path.join(self.model_dir, 'X_scaler_unified.npy'), 
                                        allow_pickle=True)
                X_scaler.mean_ = X_scaler_params[0]
                X_scaler.scale_ = X_scaler_params[1]
                
                y_scaler = StandardScaler()
                y_scaler_params = np.load(os.path.join(self.model_dir, 'y_scaler_unified.npy'), 
                                        allow_pickle=True)
                y_scaler.mean_ = y_scaler_params[0]
                y_scaler.scale_ = y_scaler_params[1]
                
                self.scalers['unified'] = {'X_scaler': X_scaler, 'y_scaler': y_scaler}
                
                print("Loaded unified model")
    
    def run_pipeline(self):
        """
        Run the complete pipeline: load data, analyze, train models
        """
        # Load data
        print("Loading data...")
        combined_df = self.load_data()
        
        # Analyze data
        print("\nAnalyzing data...")
        self.analyze_data(combined_df)
        
        # Train models
        print("\nTraining models...")
        self.train_model(combined_df)
        
        print("\nPipeline completed successfully!")
        print(f"Models and visualizations saved to '{self.model_dir}' directory")

# Main execution
if __name__ == "__main__":
    # File path to your data
    file_path = r"C:\Users\AM323\OneDrive\Desktop\Nanomaterial_Project\code1\resonancedata.xlsx"
    
    # Initialize the improved model with separate models for each refractive index
    # and physics-informed features
    model = NanophotonicModelImproved(
        data_path=file_path,
        separate_models=True,
        physics_informed=True
    )
    
    # Run the complete pipeline
    model.run_pipeline()
    
    # Example predictions
    print("\nExample predictions from trained models:")
    for ri in [1.0, 1.5]:
        for d in [50, 100, 200]:
            wavelength, extinction = model.predict(d, ri)
            print(f"Diameter: {d}nm, RI: {ri} → Wavelength: {wavelength:.2f}nm, Extinction: {extinction:.6f}")