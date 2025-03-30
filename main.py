from src.NCAA25 import logger
from src.NCAA25.pipeline.stage_01_data_ingestion import main as stage_01_main
from src.NCAA25.pipeline.stage_02_model_training import main as stage_02_main
from src.NCAA25.pipeline.stage_03_model_analysis import main as stage_03_main


if __name__ == "__main__":
    try:
        logger.info("NCAA Tournament Prediction Pipeline Started")
        #stage_01_main()
        logger.info("NCAA Tournament Prediction Pipeline Completed")

        # Run Stage 2: Model Training and Submission
        #stage_02_main()

        logger.info("NCAA Tournament Prediction Pipeline Completed")

        stage_03_main()

        logger.info("NCAA Tournament Prediction Pipeline Completed")
    except Exception as e:
        logger.error(f"Error in NCAA Tournament Prediction Pipeline: {e}")
        raise e