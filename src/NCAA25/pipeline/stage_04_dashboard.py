from src.NCAA25 import logger
from src.NCAA25.config.configuration import ConfigurationManager
from src.NCAA25.components.dashboard import Dashboard


def main():
    try:
        logger.info("Stage 4: Dashboard Creation Started")

        # Configuration
        config = ConfigurationManager()
        dashboard_config = config.get_dashboard_config()

        # Dashboard creation
        dashboard = Dashboard(config=dashboard_config)
        app_script_path = dashboard.setup_dashboard()

        logger.info("Stage 4: Dashboard Creation Completed")
        logger.info(f"Dashboard app created at: {app_script_path}")
        logger.info(f"Start dashboard with: python {app_script_path}")

        return app_script_path
    except Exception as e:
        logger.error(f"Error in Stage 4: {e}")
        raise e