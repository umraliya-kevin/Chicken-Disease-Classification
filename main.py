from ChickenDiseaseClassification import logger
from ChickenDiseaseClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ChickenDiseaseClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from ChickenDiseaseClassification.pipeline.stage_03_training import ModelTrainingPipeline
from ChickenDiseaseClassification.pipeline.stage_04_evaluation import EvaluationPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
     logger.exception(e)
     raise e


STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
     logger.exception(e)
     raise e
 
STAGE_NAME = "Training Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    training_pipeline = ModelTrainingPipeline()
    training_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
     logger.exception(e)
     raise e
 
STAGE_NAME = "Evaluation Stage"
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e