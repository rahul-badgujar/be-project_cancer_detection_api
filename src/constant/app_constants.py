class AppConstants:
    temp_directory = '/home/rahul/rahul/be-project/cancer-detection-api/temp'
    temp_file_upload_directory = f'{temp_directory}/uploads'
    temp_intermediate_files_directory = f'{temp_directory}/intermediate'
    # todo: make this path relative
    training_dataset_directory = '/home/rahul/rahul/be-project/cancer-detection-api/data/training_set'
    testing_dataset_directory = '/home/rahul/rahul/be-project/cancer-detection-api/data/testing_set'
    cancer_stage_encodings: dict = {
        '0': 'benign',
        '1': 'malignant',
        '2': 'normal'
    }
