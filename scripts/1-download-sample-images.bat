cls
poetry build 
poetry run spark-submit  --master local  --py-files dist/digital_pathology-*.whl   jobs/download.py data/0-extract.parquet  data/patient_extracts