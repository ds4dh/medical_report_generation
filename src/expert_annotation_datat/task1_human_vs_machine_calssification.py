import pandas as pd
import os


def create_classification_excel(df_human, df_machine, save_name):
    """
    Create an Excel file with human and machine-generated reports for classification.
    
    Args:
        df_human: DataFrame containing human-generated reports
        df_machine: DataFrame containing machine-generated reports
        save_name: Base name for the output file
    
    Returns:
        DataFrame with shuffled reports and metadata
    """
    output_file = f'{save_name}classification.xlsx'
    reports = []
    agents = []
    sources = []
    specialties = []
    report_types = []
    classification = []
    
    # Add human reports with their metadata
    for _, row in df_human.iterrows():
        reports.append(row['true_text'])
        agents.append("human")
        sources.append(row['source'])
        specialties.append(row.get('doctor_specilaity', ''))
        report_types.append(row.get('report_type', ''))
        classification.append('')
    
    # Add machine reports with their metadata
    for _, row in df_machine.iterrows():
        reports.append(row['generation'])
        agents.append("machine")
        sources.append(row['source'])
        specialties.append(row.get('doctor_specilaity', ''))
        report_types.append(row.get('report_type', ''))
        classification.append('')
    
    # Create DataFrame and shuffle
    df = pd.DataFrame({
        "report": reports,
        "agent": agents,
        "source": sources,
        "specialty": specialties,
        "report_type": report_types,
        'classification': classification
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Excel file created successfully: {output_file}")
    
    return df


def main(input_folder, output_folder):
    """
    Main function to process CSV files and create classification Excel files.
    
    Args:
        input_folder: Path to folder containing input CSV files
        output_folder: Path to folder where output Excel files will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get sorted CSV files containing 'score' in filename
    csv_files = sorted([file for file in os.listdir(input_folder) 
                       if file.endswith('.csv') and 'score' in file])
    
    for cv_file in csv_files:
        # Read and shuffle CSV
        df = pd.read_csv(os.path.join(input_folder, cv_file), encoding='utf-8')
        df = df.sample(frac=1).reset_index(drop=True)
        
        df_human = df.sample(n=6)
        df_machine = df.sample(n=6)
        
        # Determine language and save path
        lang = 'ENG' if 'ENG' in cv_file else 'FRE'
        save_add_name = (f'{output_folder}/Task1_case_reports_{lang}_' 
                        if 'PubMed' in cv_file 
                        else f'{output_folder}/Task1_medical_transcripts_{lang}_')
        
        create_classification_excel(df_human, df_machine, save_add_name)


if __name__ == "__main__":
    # Configure your paths here
    INPUT_FOLDER = 'path/to/your/input/folder'  # Replace with your input folder path
    OUTPUT_FOLDER = 'path/to/your/output/folder'  # Replace with your output folder path
    
    main(INPUT_FOLDER, OUTPUT_FOLDER)
