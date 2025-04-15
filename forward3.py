import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

np.random.seed(42)
tf.random.set_seed(42)

class NanophotonicCurveFittingModel:
    def __init__(self, data_path, separate_models=True, polynomial_degree=3):
        self.data_path = data_path
        self.separate_models = separate_models
        self.polynomial_degree = polynomial_degree
        self.models = {}
        self.scalers = {}
        self.poly_transformers = {}
        self.refractive_indices = [1.0, 1.2, 1.3, 1.33, 1.4, 1.5]
        self.model_dir = "curve_fitting_models"
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_data(self):
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
                df = pd.read_excel(self.data_path, sheet_name=sheet_name)
                df.columns = [col.strip().lower() for col in df.columns]
                
                column_mapping = {
                    'size (diameter)': 'diameter',
                    'refractive index of surrounding': 'refractive_index',
                    'resonance wavelength': 'resonance_wavelength',
                    'resonance intensity (extinction)': 'extinction'
                }
                
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                if 'refractive_index' not in df.columns:
                    df['refractive_index'] = ri
                
                all_data.append(df)
                print(f"Successfully loaded sheet {sheet_name} with refractive index {ri}")
                
            except Exception as e:
                print(f"Error loading sheet {sheet_name}: {e}")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print("Available columns:", combined_data.columns.tolist())
        print(f"Loaded {len(combined_data)} data points")
        
        if combined_data.isna().any().any() or np.isinf(combined_data.values).any():
            print("Warning: Dataset contains NaN or infinite values. Cleaning...")
            combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna()
            print(f"After cleaning: {len(combined_data)} data points")
        
        return combined_data
    
    def analyze_data(self, df):
        plt.figure(figsize=(12, 8))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        for ri in self.refractive_indices:
            subset = df[df['refractive_index'] == ri]
            ax1.scatter(subset['diameter'], subset['resonance_wavelength'], 
                     alpha=0.5, label=f'RI = {ri} (Wavelength)')
            
        ax1.set_xlabel('Diameter (nm)')
        ax1.set_ylabel('Resonance Wavelength (nm)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title("Resonance Wavelength and Extinction vs Diameter")
        
        for ri in self.refractive_indices:
            subset = df[df['refractive_index'] == ri]
            ax2.scatter(subset['diameter'], subset['extinction'], 
                     alpha=0.5, label=f'RI = {ri} (Extinction)')
        
        ax2.set_ylabel('Extinction', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'data_analysis.png'))
        plt.close()
        
        # Look at the trends for each refractive index
        for ri in self.refractive_indices:
            subset = df[df['refractive_index'] == ri]
            
            if len(subset) < 10:
                print(f"Insufficient data for RI = {ri}, skipping trend analysis")
                continue
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Sort by diameter for cleaner curve
            subset = subset.sort_values('diameter')
            
            # Resonance wavelength vs diameter
            ax1.plot(subset['diameter'], subset['resonance_wavelength'], 'o-')
            ax1.set_title(f'Resonance Wavelength vs Diameter (RI = {ri})')
            ax1.set_xlabel('Diameter (nm)')
            ax1.set_ylabel('Resonance Wavelength (nm)')
            ax1.grid(True)
            
            # Extinction vs diameter
            ax2.plot(subset['diameter'], subset['extinction'], 'o-', color='red')
            ax2.set_title(f'Extinction vs Diameter (RI = {ri})')
            ax2.set_xlabel('Diameter (nm)')
            ax2.set_ylabel('Extinction')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, f'trend_analysis_ri_{ri:.2f}.png'))
            plt.close()
    
    def build_model(self, input_dim):
        inputs = Input(shape=(input_dim,))
        
        # Deep branch for complex patterns
        x = Dense(64)(inputs)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        
        x = Dense(128)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        
        # Separate outputs for wavelength and extinction with appropriate activation
        wavelength_branch = Dense(32)(x)
        wavelength_branch = LeakyReLU(alpha=0.1)(wavelength_branch)
        wavelength_output = Dense(1, activation='softplus', name='wavelength')(wavelength_branch)
        
        extinction_branch = Dense(32)(x)
        extinction_branch = LeakyReLU(alpha=0.1)(extinction_branch)
        extinction_output = Dense(1, activation='softplus', name='extinction')(extinction_branch)
        
        model = Model(inputs, [wavelength_output, extinction_output])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'wavelength': 'mse', 'extinction': 'mse'},
            loss_weights={'wavelength': 1.0, 'extinction': 1.0},
            metrics={'wavelength': 'mae', 'extinction': 'mae'}
        )
        
        return model

    def prepare_data_for_training(self, df, ri=None):
        if ri is not None:
            df = df[df['refractive_index'] == ri].copy()
        else:
            df = df.copy()
        
        # Select basic features
        X_base = df[['diameter', 'refractive_index']].values
        
        # Apply polynomial transformation to capture non-linear relationships
        poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
        X_poly = poly.fit_transform(X_base)
        
        y_wavelength = df['resonance_wavelength'].values.reshape(-1, 1)
        y_extinction = df['extinction'].values.reshape(-1, 1)
        
        # Split data
        X_train, X_test, y_wavelength_train, y_wavelength_test, y_extinction_train, y_extinction_test = train_test_split(
            X_poly, y_wavelength, y_extinction, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        
        # Scale targets - use different scalers for wavelength and extinction
        wavelength_scaler = MinMaxScaler(feature_range=(0, 1))
        y_wavelength_train_scaled = wavelength_scaler.fit_transform(y_wavelength_train)
        y_wavelength_test_scaled = wavelength_scaler.transform(y_wavelength_test)
        
        extinction_scaler = MinMaxScaler(feature_range=(0, 1))
        y_extinction_train_scaled = extinction_scaler.fit_transform(y_extinction_train)
        y_extinction_test_scaled = extinction_scaler.transform(y_extinction_test)
        
        # Store min/max wavelength and extinction for bounds checking later
        wavelength_min = np.min(y_wavelength)
        extinction_min = np.min(y_extinction)
        
        return {
            'X_train_scaled': X_train_scaled, 
            'X_test_scaled': X_test_scaled,
            'y_wavelength_train_scaled': y_wavelength_train_scaled,
            'y_wavelength_test_scaled': y_wavelength_test_scaled,
            'y_extinction_train_scaled': y_extinction_train_scaled,
            'y_extinction_test_scaled': y_extinction_test_scaled,
            'X_scaler': X_scaler,
            'wavelength_scaler': wavelength_scaler,
            'extinction_scaler': extinction_scaler,
            'poly': poly,
            'wavelength_min': wavelength_min,
            'extinction_min': extinction_min,
            'X_train': X_train,
            'y_wavelength_train': y_wavelength_train,
            'y_extinction_train': y_extinction_train,
            'X_test': X_test,
            'y_wavelength_test': y_wavelength_test,
            'y_extinction_test': y_extinction_test
        }
    
    def train_model(self, combined_df):
        if self.separate_models:
            for ri in self.refractive_indices:
                print(f"\nTraining model for refractive index {ri}")
                
                subset = combined_df[combined_df['refractive_index'] == ri]
                if len(subset) < 10:
                    print(f"Insufficient data for RI = {ri}, skipping model training")
                    continue
                
                data = self.prepare_data_for_training(combined_df, ri)
                
                input_dim = data['X_train_scaled'].shape[1]
                model = self.build_model(input_dim)
                
                callbacks = [
                    EarlyStopping(patience=40, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-6, monitor='val_loss'),
                    ModelCheckpoint(
                        filepath=os.path.join(self.model_dir, f'model_ri_{ri:.2f}.h5'),
                        save_best_only=True, monitor='val_loss'
                    )
                ]
                
                history = model.fit(
                    data['X_train_scaled'], 
                    {'wavelength': data['y_wavelength_train_scaled'], 
                     'extinction': data['y_extinction_train_scaled']},
                    epochs=500,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1
                )
                
                self.plot_training_history(history, ri)
                
                y_pred = model.predict(data['X_test_scaled'])
                y_wavelength_pred = data['wavelength_scaler'].inverse_transform(y_pred[0])
                y_extinction_pred = data['extinction_scaler'].inverse_transform(y_pred[1])
                
                rmse_wavelength = np.sqrt(mean_squared_error(data['y_wavelength_test'], y_wavelength_pred))
                rmse_extinction = np.sqrt(mean_squared_error(data['y_extinction_test'], y_extinction_pred))
                r2_wavelength = r2_score(data['y_wavelength_test'], y_wavelength_pred)
                r2_extinction = r2_score(data['y_extinction_test'], y_extinction_pred)
                
                print(f"RI = {ri} - RMSE Wavelength: {rmse_wavelength:.4f}, R² Wavelength: {r2_wavelength:.4f}")
                print(f"RI = {ri} - RMSE Extinction: {rmse_extinction:.6f}, R² Extinction: {r2_extinction:.4f}")
                
                # Store the model before plotting predictions
                self.models[ri] = model
                self.scalers[ri] = {
                    'X_scaler': data['X_scaler'], 
                    'wavelength_scaler': data['wavelength_scaler'],
                    'extinction_scaler': data['extinction_scaler']
                }
                self.poly_transformers[ri] = data['poly']
                
                # Now that the model is stored, we can plot predictions
                self.plot_predictions(data, y_wavelength_pred, y_extinction_pred, ri)
                
                # Save scalers and transformer
                np.save(os.path.join(self.model_dir, f'X_scaler_ri_{ri:.2f}.npy'), 
                       [data['X_scaler'].data_min_, data['X_scaler'].data_max_, data['X_scaler'].data_range_, data['X_scaler'].scale_])
                np.save(os.path.join(self.model_dir, f'wavelength_scaler_ri_{ri:.2f}.npy'), 
                       [data['wavelength_scaler'].data_min_, data['wavelength_scaler'].data_max_, data['wavelength_scaler'].data_range_, data['wavelength_scaler'].scale_])
                np.save(os.path.join(self.model_dir, f'extinction_scaler_ri_{ri:.2f}.npy'), 
                       [data['extinction_scaler'].data_min_, data['extinction_scaler'].data_max_, data['extinction_scaler'].data_range_, data['extinction_scaler'].scale_])
                
        else:
            print("\nTraining unified model for all refractive indices")
            
            data = self.prepare_data_for_training(combined_df)
            
            input_dim = data['X_train_scaled'].shape[1]
            model = self.build_model(input_dim)
            
            callbacks = [
                EarlyStopping(patience=40, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-6, monitor='val_loss'),
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'unified_model.h5'),
                    save_best_only=True, monitor='val_loss'
                )
            ]
            
            history = model.fit(
                data['X_train_scaled'], 
                {'wavelength': data['y_wavelength_train_scaled'], 
                 'extinction': data['y_extinction_train_scaled']},
                epochs=500,
                batch_size=16,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            self.plot_training_history(history)
            
            y_pred = model.predict(data['X_test_scaled'])
            y_wavelength_pred = data['wavelength_scaler'].inverse_transform(y_pred[0])
            y_extinction_pred = data['extinction_scaler'].inverse_transform(y_pred[1])
            
            rmse_wavelength = np.sqrt(mean_squared_error(data['y_wavelength_test'], y_wavelength_pred))
            rmse_extinction = np.sqrt(mean_squared_error(data['y_extinction_test'], y_extinction_pred))
            r2_wavelength = r2_score(data['y_wavelength_test'], y_wavelength_pred)
            r2_extinction = r2_score(data['y_extinction_test'], y_extinction_pred)
            
            print(f"Unified Model - RMSE Wavelength: {rmse_wavelength:.4f}, R² Wavelength: {r2_wavelength:.4f}")
            print(f"Unified Model - RMSE Extinction: {rmse_extinction:.6f}, R² Extinction: {r2_extinction:.4f}")
            
            # Store the model before plotting predictions
            self.models['unified'] = model
            self.scalers['unified'] = {
                'X_scaler': data['X_scaler'], 
                'wavelength_scaler': data['wavelength_scaler'],
                'extinction_scaler': data['extinction_scaler']
            }
            self.poly_transformers['unified'] = data['poly']
            
            # Now that the model is stored, we can plot predictions
            self.plot_predictions(data, y_wavelength_pred, y_extinction_pred)
            
            # Save scalers and transformer
            np.save(os.path.join(self.model_dir, 'X_scaler_unified.npy'), 
                   [data['X_scaler'].data_min_, data['X_scaler'].data_max_, data['X_scaler'].data_range_, data['X_scaler'].scale_])
            np.save(os.path.join(self.model_dir, 'wavelength_scaler_unified.npy'), 
                   [data['wavelength_scaler'].data_min_, data['wavelength_scaler'].data_max_, data['wavelength_scaler'].data_range_, data['wavelength_scaler'].scale_])
            np.save(os.path.join(self.model_dir, 'extinction_scaler_unified.npy'), 
                   [data['extinction_scaler'].data_min_, data['extinction_scaler'].data_max_, data['extinction_scaler'].data_range_, data['extinction_scaler'].scale_])
        
        return self.models
    
    def plot_training_history(self, history, ri=None):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Total Loss')
        plt.plot(history.history['val_loss'], label='Val Total Loss')
        if ri is not None:
            plt.title(f'Model Loss (RI = {ri})')
        else:
            plt.title('Model Loss (Unified)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['wavelength_loss'], label='Wavelength Loss')
        plt.plot(history.history['val_wavelength_loss'], label='Val Wavelength Loss')
        if ri is not None:
            plt.title(f'Wavelength Loss (RI = {ri})')
        else:
            plt.title('Wavelength Loss (Unified)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['extinction_loss'], label='Extinction Loss')
        plt.plot(history.history['val_extinction_loss'], label='Val Extinction Loss')
        if ri is not None:
            plt.title(f'Extinction Loss (RI = {ri})')
        else:
            plt.title('Extinction Loss (Unified)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if ri is not None:
            plt.savefig(os.path.join(self.model_dir, f'training_history_ri_{ri:.2f}.png'))
        else:
            plt.savefig(os.path.join(self.model_dir, 'training_history_unified.png'))
        
        plt.close()
    
    def plot_predictions(self, data, y_wavelength_pred, y_extinction_pred, ri=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Wavelength predictions
        ax1.scatter(data['y_wavelength_test'], y_wavelength_pred, alpha=0.7)
        min_val = min(data['y_wavelength_test'].min(), y_wavelength_pred.min())
        max_val = max(data['y_wavelength_test'].max(), y_wavelength_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
        if ri is not None:
            ax1.set_title(f'Predicted vs Actual Resonance Wavelength (RI = {ri})')
        else:
            ax1.set_title('Predicted vs Actual Resonance Wavelength (Unified)')
        ax1.set_xlabel('Actual Wavelength (nm)')
        ax1.set_ylabel('Predicted Wavelength (nm)')
        ax1.grid(True)
        
        # Extinction predictions
        ax2.scatter(data['y_extinction_test'], y_extinction_pred, alpha=0.7, color='red')
        min_val = min(data['y_extinction_test'].min(), y_extinction_pred.min())
        max_val = max(data['y_extinction_test'].max(), y_extinction_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
        if ri is not None:
            ax2.set_title(f'Predicted vs Actual Extinction (RI = {ri})')
        else:
            ax2.set_title('Predicted vs Actual Extinction (Unified)')
        ax2.set_xlabel('Actual Extinction')
        ax2.set_ylabel('Predicted Extinction')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if ri is not None:
            plt.savefig(os.path.join(self.model_dir, f'predictions_ri_{ri:.2f}.png'))
        else:
            plt.savefig(os.path.join(self.model_dir, 'predictions_unified.png'))
        
        plt.close()
        
        # Also plot the curve fit
        X_base = np.array([[d, data['y_wavelength_test'][0][0]] for d in range(10, 301, 5)])
        if ri is not None:
            X_base[:, 1] = ri
        X_poly = data['poly'].transform(X_base)
        X_scaled = data['X_scaler'].transform(X_poly)
        
        model = self.models[ri] if ri is not None else self.models['unified']
        preds = model.predict(X_scaled)
        
        wavelength_preds = data['wavelength_scaler'].inverse_transform(preds[0])
        extinction_preds = data['extinction_scaler'].inverse_transform(preds[1])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot wavelength curve
        ax1.scatter(X_base[:, 0], wavelength_preds, color='blue', s=10)
        ax1.plot(X_base[:, 0], wavelength_preds, 'b-', alpha=0.7)
        if ri is not None:
            ax1.set_title(f'Resonance Wavelength vs Diameter (RI = {ri})')
        else:
            ax1.set_title('Resonance Wavelength vs Diameter (Unified)')
        ax1.set_xlabel('Diameter (nm)')
        ax1.set_ylabel('Resonance Wavelength (nm)')
        ax1.grid(True)
        
        # Plot extinction curve
        ax2.scatter(X_base[:, 0], extinction_preds, color='red', s=10)
        ax2.plot(X_base[:, 0], extinction_preds, 'r-', alpha=0.7)
        if ri is not None:
            ax2.set_title(f'Extinction vs Diameter (RI = {ri})')
        else:
            ax2.set_title('Extinction vs Diameter (Unified)')
        ax2.set_xlabel('Diameter (nm)')
        ax2.set_ylabel('Extinction')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if ri is not None:
            plt.savefig(os.path.join(self.model_dir, f'curve_fit_ri_{ri:.2f}.png'))
        else:
            plt.savefig(os.path.join(self.model_dir, 'curve_fit_unified.png'))
        
        plt.close()
    
    def predict(self, diameter, refractive_index):
        if diameter <= 0:
            raise ValueError("Diameter must be positive")
            
        if self.separate_models:
            closest_ri = min(self.refractive_indices, key=lambda x: abs(x - refractive_index))
            if abs(closest_ri - refractive_index) > 0.01:
                print(f"Using closest available RI: {closest_ri}")
            
            model = self.models[closest_ri]
            X_scaler = self.scalers[closest_ri]['X_scaler']
            wavelength_scaler = self.scalers[closest_ri]['wavelength_scaler']
            extinction_scaler = self.scalers[closest_ri]['extinction_scaler']
            poly = self.poly_transformers[closest_ri]
            
            X_base = np.array([[diameter, closest_ri]])
            X_poly = poly.transform(X_base)
            X_scaled = X_scaler.transform(X_poly)
        else:
            model = self.models['unified']
            X_scaler = self.scalers['unified']['X_scaler']
            wavelength_scaler = self.scalers['unified']['wavelength_scaler']
            extinction_scaler = self.scalers['unified']['extinction_scaler']
            poly = self.poly_transformers['unified']
            
            X_base = np.array([[diameter, refractive_index]])
            X_poly = poly.transform(X_base)
            X_scaled = X_scaler.transform(X_poly)
        
        preds = model.predict(X_scaled)
        
        wavelength_pred = wavelength_scaler.inverse_transform(preds[0])[0][0]
        extinction_pred = extinction_scaler.inverse_transform(preds[1])[0][0]
        
        # Force minimum physically meaningful values
        wavelength_pred = max(300, wavelength_pred)  # Lowest physically possible wavelength for resonance
        extinction_pred = max(0, extinction_pred)  # Extinction must be non-negative
        
        return wavelength_pred, extinction_pred
    
    def load_saved_models(self):
        if self.separate_models:
            for ri in self.refractive_indices:
                model_path = os.path.join(self.model_dir, f'model_ri_{ri:.2f}.h5')
                if os.path.exists(model_path):
                    try:
                        self.models[ri] = tf.keras.models.load_model(model_path)
                        
                        # Load scalers
                        X_scaler = MinMaxScaler()
                        X_scaler_params = np.load(os.path.join(self.model_dir, f'X_scaler_ri_{ri:.2f}.npy'), 
                                                allow_pickle=True)
                        X_scaler.data_min_ = X_scaler_params[0]
                        X_scaler.data_max_ = X_scaler_params[1]
                        X_scaler.data_range_ = X_scaler_params[2]
                        X_scaler.scale_ = X_scaler_params[3]
                        
                        wavelength_scaler = MinMaxScaler()
                        wavelength_scaler_params = np.load(os.path.join(self.model_dir, f'wavelength_scaler_ri_{ri:.2f}.npy'), 
                                                        allow_pickle=True)
                        wavelength_scaler.data_min_ = wavelength_scaler_params[0]
                        wavelength_scaler.data_max_ = wavelength_scaler_params[1]
                        wavelength_scaler.data_range_ = wavelength_scaler_params[2]
                        wavelength_scaler.scale_ = wavelength_scaler_params[3]
                        
                        extinction_scaler = MinMaxScaler()
                        extinction_scaler_params = np.load(os.path.join(self.model_dir, f'extinction_scaler_ri_{ri:.2f}.npy'), 
                                                        allow_pickle=True)
                        extinction_scaler.data_min_ = extinction_scaler_params[0]
                        extinction_scaler.data_max_ = extinction_scaler_params[1]
                        extinction_scaler.data_range_ = extinction_scaler_params[2]
                        extinction_scaler.scale_ = extinction_scaler_params[3]
                        
                        self.scalers[ri] = {
                            'X_scaler': X_scaler, 
                            'wavelength_scaler': wavelength_scaler,
                            'extinction_scaler': extinction_scaler
                        }
                        
                        # Recreate polynomial transformer
                        df = pd.read_excel(self.data_path, sheet_name=str(self.refractive_indices.index(ri) + 1))
                        X_base = df[['diameter', 'refractive_index']].values
                        poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
                        poly.fit(X_base)
                        self.poly_transformers[ri] = poly
                        
                        print(f"Successfully loaded model for RI = {ri}")
                    except Exception as e:
                        print(f"Error loading model for RI = {ri}: {e}")
        else:
            model_path = os.path.join(self.model_dir, 'unified_model.h5')
            if os.path.exists(model_path):
                try:
                    self.models['unified'] = tf.keras.models.load_model(model_path)
                    
                    # Load scalers
                    X_scaler = MinMaxScaler()
                    X_scaler_params = np.load(os.path.join(self.model_dir, 'X_scaler_unified.npy'), 
                                             allow_pickle=True)
                    X_scaler.data_min_ = X_scaler_params[0]
                    X_scaler.data_max_ = X_scaler_params[1]
                    X_scaler.data_range_ = X_scaler_params[2]
                    X_scaler.scale_ = X_scaler_params[3]
                    
                    wavelength_scaler = MinMaxScaler()
                    wavelength_scaler_params = np.load(os.path.join(self.model_dir, 'wavelength_scaler_unified.npy'), 
                                                     allow_pickle=True)
                    wavelength_scaler.data_min_ = wavelength_scaler_params[0]
                    wavelength_scaler.data_max_ = wavelength_scaler_params[1]
                    wavelength_scaler.data_range_ = wavelength_scaler_params[2]
                    wavelength_scaler.scale_ = wavelength_scaler_params[3]
                    
                    extinction_scaler = MinMaxScaler()
                    extinction_scaler_params = np.load(os.path.join(self.model_dir, 'extinction_scaler_unified.npy'), 
                                                     allow_pickle=True)
                    extinction_scaler.data_min_ = extinction_scaler_params[0]
                    extinction_scaler.data_max_ = extinction_scaler_params[1]
                    extinction_scaler.data_range_ = extinction_scaler_params[2]
                    extinction_scaler.scale_ = extinction_scaler_params[3]
                    
                    self.scalers['unified'] = {
                        'X_scaler': X_scaler, 
                        'wavelength_scaler': wavelength_scaler,
                        'extinction_scaler': extinction_scaler
                    }
                    
                    # Recreate polynomial transformer
                    df = pd.read_excel(self.data_path)
                    X_base = df[['diameter', 'refractive_index']].values
                    poly = PolynomialFeatures(degree=self.polynomial_degree, include_bias=False)
                    poly.fit(X_base)
                    self.poly_transformers['unified'] = poly
                    
                    print("Successfully loaded unified model")
                except Exception as e:
                    print(f"Error loading unified model: {e}")
    
    def run(self):
        df = self.load_data()
        self.analyze_data(df)
        
        try:
            self.load_saved_models()
            print("Successfully loaded existing models")
        except Exception as e:
            print(f"Error loading existing models: {e}")
            print("Training new models...")
            self.train_model(df)
        
        return self.models
    
    def generate_prediction_grid(self, diameter_range=(50, 300), ri_range=(1.0, 1.5), diameter_step=10, ri_step=0.1):
        diameters = np.arange(diameter_range[0], diameter_range[1] + diameter_step, diameter_step)
        ris = np.arange(ri_range[0], ri_range[1] + ri_step, ri_step)
        
        grid_results = []
        
        for ri in ris:
            wavelengths = []
            extinctions = []
            
            for d in diameters:
                wavelength, extinction = self.predict(d, ri)
                wavelengths.append(wavelength)
                extinctions.append(extinction)
            
            grid_results.append({
                'refractive_index': ri,
                'diameters': diameters,
                'wavelengths': wavelengths,
                'extinctions': extinctions
            })
        
        # Create plots
        plt.figure(figsize=(16, 10))
        
        # Wavelength vs Diameter for different RIs
        plt.subplot(2, 1, 1)
        for result in grid_results:
            plt.plot(result['diameters'], result['wavelengths'], 
                     marker='o', label=f"RI = {result['refractive_index']:.2f}")
        
        plt.xlabel('Diameter (nm)')
        plt.ylabel('Resonance Wavelength (nm)')
        plt.title('Predicted Resonance Wavelength vs Nanoparticle Diameter')
        plt.grid(True)
        plt.legend()
        
        # Extinction vs Diameter for different RIs
        plt.subplot(2, 1, 2)
        for result in grid_results:
            plt.plot(result['diameters'], result['extinctions'], 
                     marker='o', label=f"RI = {result['refractive_index']:.2f}")
        
        plt.xlabel('Diameter (nm)')
        plt.ylabel('Extinction')
        plt.title('Predicted Extinction vs Nanoparticle Diameter')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'prediction_grid.png'))
        plt.close()
        
        return grid_results
    
    def plot_3d_prediction_surface(self, diameter_range=(50, 300), ri_range=(1.0, 1.5), diameter_step=10, ri_step=0.05):
        diameters = np.arange(diameter_range[0], diameter_range[1] + diameter_step, diameter_step)
        ris = np.arange(ri_range[0], ri_range[1] + ri_step, ri_step)
        
        # Create meshgrid
        D, R = np.meshgrid(diameters, ris)
        
        # Initialize arrays for predictions
        W = np.zeros_like(D, dtype=float)
        E = np.zeros_like(D, dtype=float)
        
        # Fill with predictions
        for i in range(len(ris)):
            for j in range(len(diameters)):
                W[i, j], E[i, j] = self.predict(diameters[j], ris[i])
        
        # Plot surfaces
        fig = plt.figure(figsize=(15, 10))
        
        # Wavelength surface
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        surf1 = ax1.plot_surface(D, R, W, cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.set_xlabel('Diameter (nm)')
        ax1.set_ylabel('Refractive Index')
        ax1.set_zlabel('Resonance Wavelength (nm)')
        ax1.set_title('3D Prediction Surface: Resonance Wavelength')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Wavelength (nm)')
        
        # Extinction surface
        ax2 = fig.add_subplot(2, 1, 2, projection='3d')
        surf2 = ax2.plot_surface(D, R, E, cmap='plasma', alpha=0.8, edgecolor='none')
        ax2.set_xlabel('Diameter (nm)')
        ax2.set_ylabel('Refractive Index')
        ax2.set_zlabel('Extinction')
        ax2.set_title('3D Prediction Surface: Extinction')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label='Extinction')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, '3d_prediction_surface.png'))
        plt.close()
        
        return D, R, W, E

# Example usage
if __name__ == "__main__":
    model = NanophotonicCurveFittingModel(data_path="resonancedata.xlsx", separate_models=True)
    models = model.run()
    
    # Generate prediction grid
    grid_results = model.generate_prediction_grid()
    
    # Generate 3D prediction surface
    D, R, W, E = model.plot_3d_prediction_surface()
    
    # Specific predictions
    wavelength, extinction = model.predict(diameter=100, refractive_index=1.33)
    print(f"\nPrediction for nanoparticle with diameter 100nm in water (RI=1.33):")
    print(f"Resonance Wavelength: {wavelength:.2f} nm")
    print(f"Extinction: {extinction:.4f}")