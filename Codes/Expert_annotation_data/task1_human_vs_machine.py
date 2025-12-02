import json
import csv
import os
import random
import pandas as pd


def create_classification_excel(df_human, df_machine, save_name): 

    output_file=f'{save_name}classification.xlsx'
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
        specialties.append(row['doctor_specilaity'] if 'doctor_specilaity' in row else '')
        report_types.append(row['report_type'] if 'report_type' in row else '')
        classification.append('')

        
    # Add machine reports with their metadata
    for _, row in df_machine.iterrows():
        reports.append(row['generation'])
        agents.append("machine")
        sources.append(row['source'])
        specialties.append(row['doctor_specilaity'] if 'doctor_specilaity' in row else '')
        report_types.append(row['report_type'] if 'report_type' in row else '')
        classification.append('')

    df = pd.DataFrame({
        "report": reports,
        "agent": agents,
        "source": sources,
        "specialty": specialties,
        "report_type": report_types,
        'classification':classification
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Excel file created successfully: {output_file}")
    
    return df


folder_path = '/Users/rouhizad/Documents/PycharmProjects/emerge-llm/code/Annotation/Journal/Phi4_outputs'
save_add = '/Users/rouhizad/Documents/PycharmProjects/emerge-llm/code/Annotation/Journal/Task1'
if not os.path.exists(save_add):
    os.mkdir(save_add)
    
csv_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv') and 'score' in file])
for cv_file in csv_files:

    df  = pd.read_csv(f'{folder_path}/{cv_file}', encoding='utf-8')
    df = df.sample(frac=1).reset_index(drop=True)

    df_human = df.sample(n=10)
    df_machine = df.sample(n=10)
    lang = 'ENG' if 'ENG' in cv_file else 'FRE'
    save_add_name = f'{save_add}/Task1_case_reports_{lang}_' if 'PubMed' in cv_file else f'{save_add}/Task1_medical_transcripts_{lang}_'
    

    create_classification_excel(df_human, df_machine, save_add_name) 
