from src.NCAA25 import logger
from src.NCAA25.pipeline.stage_01_data_ingestion import main as stage_01_main

if __name__ == "__main__":
    try:
        logger.info("NCAA Tournament Prediction Pipeline Started")
        stage_01_main()
        logger.info("NCAA Tournament Prediction Pipeline Completed")
    except Exception as e:
        logger.error(f"Error in NCAA Tournament Prediction Pipeline: {e}")
        raise e