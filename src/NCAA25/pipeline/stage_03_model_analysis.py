from src.NCAA25 import logger
from src.NCAA25.config.configuration import ConfigurationManager
from src.NCAA25.components.model_analysis import ModelAnalysis


def main():
    try:
        logger.info("Stage 3: Model Analysis Started")

        # Configuration
        config = ConfigurationManager()
        model_analysis_config = config.get_model_analysis_config()

        # Model Analysis
        model_analysis = ModelAnalysis(config=model_analysis_config)
        analysis_results = model_analysis.run_analysis()

        logger.info("Stage 3: Model Analysis Completed")
        return analysis_results
    except Exception as e:
        logger.error(f"Error in Stage 3: {e}")
        raise e