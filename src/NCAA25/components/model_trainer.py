import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, mean_absolute_error, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from src.NCAA25 import logger
from src.NCAA25.entity import ModelTrainerConfig
from src.NCAA25.utils.common import load_model, save_model, load_dataframe


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = None
        self.calibrator = None

    def load_train_data(self):
        """
        Load the training data
        """
        try:
            # Load training data
            train_data = load_dataframe(self.config.train_data_path)
            logger.info(f"Loaded training data: {train_data.shape}")

            # Load feature columns
            feature_cols = pd.read_csv(self.config.feature_columns_file)
            features = feature_cols['feature'].tolist()
            logger.info(f"Loaded {len(features)} feature columns")

            return train_data, features
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise e

    def load_preprocessors(self):
        """
        Load the imputer and scaler
        """
        try:
            imputer_path = os.path.join(os.path.dirname(self.config.train_data_path), "imputer.joblib")
            scaler_path = os.path.join(os.path.dirname(self.config.train_data_path), "scaler.joblib")

            imputer = load_model(imputer_path)
            scaler = load_model(scaler_path)

            logger.info("Loaded preprocessors")

            return imputer, scaler
        except Exception as e:
            logger.error(f"Error loading preprocessors: {e}")
            raise e

    def train(self):
        """
        Train the model
        """
        try:
            # Load data and preprocessors
            train_data, features = self.load_train_data()
            imputer, scaler = self.load_preprocessors()

            # Prepare features and target
            X = train_data[features].fillna(-1)
            y = train_data['Pred']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            # Preprocess data
            X_train_imputed = imputer.transform(X_train)
            X_train_scaled = scaler.transform(X_train_imputed)

            X_test_imputed = imputer.transform(X_test)
            X_test_scaled = scaler.transform(X_test_imputed)

            # Initialize and train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=self.config.model_params.n_estimators,
                max_depth=self.config.model_params.max_depth,
                min_samples_split=self.config.model_params.min_samples_split,
                max_features=self.config.model_params.max_features,
                random_state=self.config.model_params.random_state
            )

            self.model.fit(X_train_scaled, y_train)

            # Make predictions on test set
            y_pred = self.model.predict(X_test_scaled).clip(0.001, 0.999)

            # Calibrate predictions
            if self.config.calibration_method == 'isotonic':
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
                self.calibrator.fit(y_pred, y_test)
                y_pred_cal = self.calibrator.transform(y_pred)
            else:
                y_pred_cal = y_pred

            # Calculate evaluation metrics
            brier = brier_score_loss(y_test, y_pred_cal)
            log_l = log_loss(y_test, y_pred_cal)
            mae = mean_absolute_error(y_test, y_pred_cal)

            logger.info(f"Model training completed with metrics:")
            logger.info(f"Brier Score: {brier:.4f}")
            logger.info(f"Log Loss: {log_l:.4f}")
            logger.info(f"MAE: {mae:.4f}")

            # Cross-validation for more robust evaluation
            cv_scores = cross_val_score(
                self.model, X, y,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            logger.info(f"Cross-validated MSE: {-cv_scores.mean():.4f}")

            # Save model and calibrator
            model_path = self.config.trained_model_path
            save_model(self.model, model_path)

            calibrator_path = os.path.join(
                os.path.dirname(model_path),
                "calibrator.joblib"
            )
            save_model(self.calibrator, calibrator_path)

            logger.info(f"Model saved to {model_path}")

            return brier, log_l, mae
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise e

    def create_submission(self, output_path):
        """
        Create submission file
        """
        try:
            # Generate a submission template
            template_path = os.path.join(os.path.dirname(output_path), "submission_template.csv")
            self.generate_submission_template(template_path)

            # Load submission template
            sub_data = pd.read_csv(template_path)
            logger.info(f"Using submission template with {len(sub_data)} matchups")

            # Rest of the method remains the same
            # Load feature columns and preprocessors
            _, features = self.load_train_data()
            imputer, scaler = self.load_preprocessors()

            # Extract team IDs from ID column
            sub_data['Season'] = sub_data['ID'].apply(lambda x: int(x.split('_')[0]))
            sub_data['Team1'] = sub_data['ID'].apply(lambda x: x.split('_')[1])
            sub_data['Team2'] = sub_data['ID'].apply(lambda x: x.split('_')[2])
            sub_data['IDTeams'] = sub_data.apply(lambda r: f"{r['Team1']}_{r['Team2']}", axis=1)

            # Add necessary columns to match training data
            # Set defaults for required columns
            sub_data['WLoc'] = 3  # Neutral location
            sub_data['SeedDiff'] = 0  # Default seed difference

            # Create placeholder values for all required features
            for feature in features:
                if feature not in sub_data.columns:
                    sub_data[feature] = 0

            # Prepare features
            X_sub = sub_data[features].fillna(-1)
            X_sub_imputed = imputer.transform(X_sub)
            X_sub_scaled = scaler.transform(X_sub_imputed)

            # Make predictions
            preds = self.model.predict(X_sub_scaled).clip(0.01, 0.99)

            # Calibrate predictions
            if self.calibrator is not None:
                preds_cal = self.calibrator.transform(preds)
            else:
                preds_cal = preds

            # Create submission file
            submission = pd.DataFrame({
                'ID': sub_data['ID'],
                'Pred': preds_cal
            })

            submission.to_csv(output_path, index=False)
            logger.info(f"Submission file created at {output_path}")

            return output_path
        except Exception as e:
            logger.error(f"Error creating submission: {e}")
            raise e

    def generate_submission_template(self, output_path):
        """
        Generate a submission template for the 2025 tournament
        """
        try:
            # Load teams data
            processed_dir = os.path.dirname(self.config.train_data_path)
            teams_path = os.path.join(processed_dir, "teams_processed.csv")

            if os.path.exists(teams_path):
                teams_df = pd.read_csv(teams_path)
                logger.info(f"Loaded {len(teams_df)} teams for submission template")
            else:
                # If teams file doesn't exist, try to get team IDs from the training data
                train_data = pd.read_csv(self.config.train_data_path)
                team_ids = set()
                if 'Team1' in train_data.columns:
                    team_ids.update(train_data['Team1'].unique())
                if 'Team2' in train_data.columns:
                    team_ids.update(train_data['Team2'].unique())

                teams_df = pd.DataFrame({'TeamID': sorted(list(team_ids))})
                logger.info(f"Generated teams list with {len(teams_df)} teams")

            # Generate all possible matchups for 2025
            team_ids = teams_df['TeamID'].unique()
            matchups = []

            for i, team1 in enumerate(team_ids):
                for team2 in team_ids[i + 1:]:
                    # Ensure team1 ID is lower than team2 ID
                    t1, t2 = sorted([team1, team2])
                    matchups.append({
                        'ID': f"2025_{t1}_{t2}",
                        'Pred': 0.5  # Default prediction is 50%
                    })

            # Create submission dataframe
            submission = pd.DataFrame(matchups)

            # Save to CSV
            submission.to_csv(output_path, index=False)
            logger.info(f"Generated submission template with {len(submission)} matchups")

            return output_path
        except Exception as e:
            logger.error(f"Error generating submission template: {e}")
            raise e