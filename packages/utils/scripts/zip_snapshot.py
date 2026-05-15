import utils.consol_log as log
from utils.zip_db import create_mapped_zip

logger = log.get_logger(level=log.INFO)

if __name__ == "__main__":

    config_file = "./packages/utils/scripts/snap_db_zip_conf.yaml"
    logger.info(f"Using config settings from {config_file} to zip the snapshot")
    create_mapped_zip("./packages/utils/scripts/snap_db_zip_conf.yaml")
