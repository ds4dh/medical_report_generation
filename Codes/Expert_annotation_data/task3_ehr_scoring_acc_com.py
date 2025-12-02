import json
import csv
import os
import random
random.seed(35)
import pandas as pd


def create_classification_csv(df_human, save_name): 
                            

    output_file=f'{save_name}evaluation.xlsx'

    entities = []
    reports = []
    agents = []
    sources = []
    specialties = []
    report_types = []
    accuracies = []
    fluencies = []
    compeltness =[]

    for _, row in df_human.iterrows():

        ehr_simulated = row['input_entities']

        lines = ehr_simulated.split('\n')

        # Remove first and last line
        middle_lines = lines[1:-1]

        # Join back into a string
        result = '\n'.join(middle_lines)
        
        entities.append(result)
        reports.append(row['true_text'])
        agents.append("human")
        sources.append(row['source'])
        specialties.append(row['specialty'])
        report_types.append(row['report_type'])
        
        accuracies.append('')
        fluencies.append('')
        compeltness.append('')

        
    df = pd.DataFrame({
        "Report": reports,
        'Generated EHR': entities,
        "Category": sources,
        "Specialty": specialties,
        "Report type": report_types,
        'Accuracy': accuracies, 
        'Completeness': compeltness,
    })
    df = df.sample(frac=1, random_state=30).reset_index(drop=True)

    
    df.to_excel(output_file, index=False)
    print(f"Excel file created successfully: {output_file}")
    

    return df


folder_path = '/Users/rouhizad/Documents/PycharmProjects/emerge-llm/code/Annotation/Journal/Phi4_outputs'
save_add = '/Users/rouhizad/Documents/PycharmProjects/emerge-llm/code/Annotation/Journal/Task3'

csv_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv') and 'score' in file])
print(csv_files)

for cv_file in csv_files:

    df  = pd.read_csv(f'{folder_path}/{cv_file}', encoding='utf-8')
    df = df.sample(frac=1).reset_index(drop=True)

    df_human = df.sample(n=25)
    lang = 'ENG' if 'ENG' in cv_file else 'FRE'
    save_add_name = f'{save_add}/Task3_case_reports_{lang}_' if 'PubMed' in cv_file else f'{save_add}/Task3_medical_transcripts_{lang}_'


    create_classification_csv(df_human, save_add_name)

