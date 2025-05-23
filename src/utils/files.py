import yaml
import os
import csv

def get_hyperparameters(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        hyperparameters = yaml.load(f.read(), Loader=yaml.FullLoader)
    return hyperparameters

def write_to_csv(file_path, headers, data):

    dir_name = os.path.dirname(file_path)
    
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    file_exists = os.path.isfile(file_path)

    # Check if data contains all the keys in headers
    for header in headers:
        if header not in data:
            raise ValueError(f"Missing required fields: {header}")

    with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    # file_exists = os.path.isfile(file_path)

    # # Check if data contains all the keys in headers
    # for header in headers:
    #     if header not in data:
    #         raise ValueError(f"Missing required fields: {header}")

    # with open(file_path, mode='a', newline='', encoding='utf-8') as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=headers)
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow(data)
