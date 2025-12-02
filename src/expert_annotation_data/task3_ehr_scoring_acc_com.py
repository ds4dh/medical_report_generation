import pandas as pd
import os
import random

random.seed(42)


def create_evaluation_excel(df_human, save_name):
    """
    Create an Excel file with human-generated reports and extracted EHR data for evaluation.
    
    Args:
        df_human: DataFrame containing human-generated reports
        save_name: Base name for the output file
    
    Returns:
        DataFrame with shuffled reports and metadata
    """
    output_file = f'{save_name}evaluation.xlsx'
    
    # Create empty lists to store all data
    entities = []
    reports = []
    agents = []
    sources = []
    specialties = []
    report_types = []
    accuracies = []
    fluencies = []
    completeness = []
    
    # Add human reports with their metadata
    for _, row in df_human.iterrows():
        ehr_simulated = row['input_entities']
        
        # Remove first and last line from EHR entities
        lines = ehr_simulated.split('\n')
        middle_lines = lines[1:-1]
        result = '\n'.join(middle_lines)
        
        entities.append(result)
        reports.append(row['true_text'])
        agents.append("human")
        sources.append(row['source'])
        specialties.append(row['specialty'])
        report_types.append(row['report_type'])
        accuracies.append('')
        fluencies.append('')
        completeness.append('')
    
    # Create DataFrame and shuffle
    df = pd.DataFrame({
        "Report": reports,
        'Generated EHR': entities,
        "Category": sources,
        "Specialty": specialties,
        "Report type": report_types,
        'Accuracy': accuracies,
        'Completeness': completeness,
    })
    df = df.sample(frac=1, random_state=30).reset_index(drop=True)
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Excel file created successfully: {output_file}")
    
    return df


def main(input_folder, output_folder, n_samples=n_s):
    """
    Main function to process CSV files and create evaluation Excel files.
    
    Args:
        input_folder: Path to folder containing input CSV files
        output_folder: Path to folder where output Excel files will be saved
        n_samples: Number of human reports to sample (default: n_s)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get sorted CSV files containing 'score' in filename
    csv_files = sorted([file for file in os.listdir(input_folder)
                       if file.endswith('.csv') and 'score' in file])
    
    print(f"Found CSV files: {csv_files}")
    
    for cv_file in csv_files:
        # Read and shuffle CSV
        df = pd.read_csv(os.path.join(input_folder, cv_file), encoding='utf-8')
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Sample human reports
        df_human = df.sample(n=n_samples)
        
        # Determine language and save path
        lang = 'ENG' if 'ENG' in cv_file else 'FRE'
        save_add_name = (f'{output_folder}/Task3_case_reports_{lang}_'
                        if 'PubMed' in cv_file
                        else f'{output_folder}/Task3_medical_transcripts_{lang}_')
        
        create_evaluation_excel(df_human, save_add_name)


if __name__ == "__main__":
    # Configure your paths here
    INPUT_FOLDER = 'path/to/your/input/folder'  # Replace with your input folder path
    OUTPUT_FOLDER = 'path/to/your/output/folder'  # Replace with your output folder path
    
    # Optional: customize sample size
    N_SAMPLES = n_s
    
    main(INPUT_FOLDER, OUTPUT_FOLDER, N_SAMPLES)
