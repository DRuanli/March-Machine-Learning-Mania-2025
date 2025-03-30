from src.NCAA25 import logger
from src.NCAA25.config.configuration import ConfigurationManager
from src.NCAA25.components.model_trainer import ModelTrainer
import os


def main():
    try:
        logger.info("Stage 2: Model Training Started")

        # Configuration
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        # Model Training
        model_trainer = ModelTrainer(config=model_trainer_config)
        brier, log_loss, mae = model_trainer.train()

        # Create submission file
        output_path = os.path.join(model_trainer_config.root_dir, "submission.csv")
        model_trainer.create_submission(output_path)

        logger.info("Stage 2: Model Training Completed")
        return brier, log_loss, mae
    except Exception as e:
        logger.error(f"Error in Stage 2: {e}")
        raise e