# RUN INFERENCE WITH AP MODEL - THIS WILL GENERATE GRADCAMS
python test.py -cfg ./config/mwe_config_ap.yaml

# RUN INFERENCE WITH LAT MODEL - THIS WILL GENERATE GRADCAMS
python test.py -cfg ./config/mwe_config_lat.yaml

# RUN INFERENCE WITH ENSEMBLE MODEL (AP + LAT)
python test_ensemble.py -cfg ./config/mwe_config_ensemble.yaml