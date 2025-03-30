import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, roc_curve, roc_auc_score, precision_recall_curve
from src.NCAA25 import logger
from src.NCAA25.entity import ModelAnalysisConfig
from src.NCAA25.utils.common import load_model, load_dataframe, save_dataframe


class ModelAnalysis:
    def __init__(self, config: ModelAnalysisConfig):
        self.config = config
        self.model = load_model(self.config.model_path)
        self.calibrator = load_model(self.config.calibrator_path)

    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        try:
            # Load feature columns
            feature_cols = pd.read_csv(self.config.feature_columns_file)
            features = feature_cols['feature'].tolist()

            # Get feature importances from model
            importances = self.model.feature_importances_

            # Create DataFrame for better visualization
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            })

            # Sort by importance
            feat_imp = feat_imp.sort_values('Importance', ascending=False)

            # Save feature importance table
            feat_imp.to_csv(os.path.join(self.config.analysis_reports_dir, 'feature_importance.csv'), index=False)

            # Get top 20 features for visualization
            top_features = feat_imp.head(20)

            # Create visualization
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.analysis_reports_dir, 'feature_importance.png'))

            logger.info(f"Generated feature importance analysis with {len(feat_imp)} features")

            return feat_imp
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            raise e

    def analyze_tournament_predictions(self):
        """Analyze predictions for tournament games"""
        try:
            # Load training data
            train_data = load_dataframe(self.config.train_data_path)

            # Load feature columns
            feature_cols = pd.read_csv(self.config.feature_columns_file)
            features = feature_cols['feature'].tolist()

            # Load preprocessors from the same directory as training data
            preprocessors_dir = os.path.dirname(self.config.train_data_path)
            imputer = load_model(os.path.join(preprocessors_dir, "imputer.joblib"))
            scaler = load_model(os.path.join(preprocessors_dir, "scaler.joblib"))

            # Prepare features and make predictions
            X = train_data[features].fillna(-1)
            X_imputed = imputer.transform(X)
            X_scaled = scaler.transform(X_imputed)
            y_true = train_data['Pred']

            # Generate predictions
            y_pred = self.model.predict(X_scaled).clip(0.001, 0.999)
            y_pred_cal = self.calibrator.transform(y_pred)

            # Add predictions to data
            analysis_df = train_data.copy()
            analysis_df['RawPrediction'] = y_pred
            analysis_df['CalibratedPrediction'] = y_pred_cal

            # Calculate metrics
            brier = brier_score_loss(y_true, y_pred_cal)

            # Calculate error (actual - predicted)
            analysis_df['Error'] = y_true - y_pred_cal
            analysis_df['AbsError'] = np.abs(analysis_df['Error'])

            # Analyze seed-based performance
            seed_performance = analysis_df.groupby(['Team1Seed', 'Team2Seed']).agg({
                'Pred': 'mean',
                'CalibratedPrediction': 'mean',
                'Error': ['mean', 'std'],
                'AbsError': 'mean'
            }).reset_index()

            seed_performance.columns = ['Team1Seed', 'Team2Seed', 'ActualWinRate', 'PredictedWinRate',
                                        'MeanError', 'StdError', 'MeanAbsError']

            # Save analysis results
            save_dataframe(seed_performance, os.path.join(self.config.analysis_reports_dir, 'seed_performance.csv'))
            save_dataframe(analysis_df, os.path.join(self.config.analysis_reports_dir, 'prediction_analysis.csv'))

            # Create upset probability chart
            self._create_upset_probability_chart(seed_performance)

            # ROC and Precision-Recall curves
            self._create_performance_curves(y_true, y_pred_cal)

            logger.info(f"Generated tournament prediction analysis with Brier score: {brier:.4f}")

            return analysis_df, seed_performance
        except Exception as e:
            logger.error(f"Error analyzing tournament predictions: {e}")
            raise e

    def _create_upset_probability_chart(self, seed_performance):
        """Create chart showing upset probabilities by seed difference"""
        try:
            # Calculate seed difference and group
            seed_perf = seed_performance.copy()
            seed_perf['SeedDiff'] = seed_perf['Team2Seed'] - seed_perf['Team1Seed']

            # Group by seed difference
            upset_probs = seed_perf.groupby('SeedDiff').agg({
                'ActualWinRate': 'mean',
                'PredictedWinRate': 'mean'
            }).reset_index()

            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.plot(upset_probs['SeedDiff'], upset_probs['ActualWinRate'], 'bo-', label='Actual Win Rate')
            plt.plot(upset_probs['SeedDiff'], upset_probs['PredictedWinRate'], 'ro-', label='Predicted Win Rate')
            plt.axhline(y=0.5, color='gray', linestyle='--')
            plt.xlabel('Seed Difference (Higher Seed - Lower Seed)')
            plt.ylabel('Win Probability (Lower Seed Winning)')
            plt.title('Win Probabilities by Seed Difference')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.config.analysis_reports_dir, 'upset_probability.png'))

            logger.info("Generated upset probability chart")
        except Exception as e:
            logger.error(f"Error creating upset probability chart: {e}")

    def _create_performance_curves(self, y_true, y_pred):
        """Create ROC and Precision-Recall curves"""
        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(os.path.join(self.config.analysis_reports_dir, 'roc_curve.png'))

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)

            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, label='Precision-Recall Curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig(os.path.join(self.config.analysis_reports_dir, 'precision_recall_curve.png'))

            logger.info("Generated performance curves")
        except Exception as e:
            logger.error(f"Error creating performance curves: {e}")

    def create_bracket_simulation(self):
        """Create a simulation of tournament bracket outcomes"""
        try:
            # Load submission file
            submission = pd.read_csv(self.config.submission_path)

            # Create a sample of teams (e.g., top 64 seeds)
            # For demonstration, we'll just create a random simulation

            # Create a bracket results DataFrame
            bracket_results = pd.DataFrame({
                'Seed': range(1, 17),
                'Region1': [f"Team{i}" for i in range(1, 17)],
                'Region2': [f"Team{i + 16}" for i in range(1, 17)],
                'Region3': [f"Team{i + 32}" for i in range(1, 17)],
                'Region4': [f"Team{i + 48}" for i in range(1, 17)]
            })

            # Save bracket simulation
            save_dataframe(bracket_results, os.path.join(self.config.analysis_reports_dir, 'bracket_simulation.csv'))

            logger.info("Generated bracket simulation")

            return bracket_results
        except Exception as e:
            logger.error(f"Error creating bracket simulation: {e}")
            raise e

    def run_analysis(self):
        """Run all analysis methods"""
        try:
            # Analyze feature importance
            feature_importance = self.analyze_feature_importance()

            # Analyze tournament predictions
            prediction_analysis, seed_performance = self.analyze_tournament_predictions()

            # Create bracket simulation
            bracket_simulation = self.create_bracket_simulation()

            logger.info("Completed all model analysis tasks")

            return {
                'feature_importance': feature_importance,
                'prediction_analysis': prediction_analysis,
                'seed_performance': seed_performance,
                'bracket_simulation': bracket_simulation
            }
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            raise e