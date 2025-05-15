import pandas as pd
import glob
import os

# Base directory containing metric result subfolders
base_dir = '/egr/research-actionlab/anhdao/ZenAI/metric_results'

# Iterate through each subfolder
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # List all .xlsx files in the subfolder
        xlsx_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
        dataframes = []
        
        # Read each Excel file and add a source column
        for file in xlsx_files:
            df = pd.read_excel(file)
            df['source_file'] = os.path.basename(file)
            dataframes.append(df)
        
        # Concatenate and write to a new Excel file
        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True)
            output_path = os.path.join(base_dir, f'{folder}.xlsx')
            merged_df.to_excel(output_path, index=False)
            print(f'Merged {len(xlsx_files)} files into {output_path}')
