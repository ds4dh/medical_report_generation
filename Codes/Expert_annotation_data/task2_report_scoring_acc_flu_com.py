import json
import csv
import os
import random
random.seed(42)
import pandas as pd


def create_classification_csv(df_human, df_machine, save_name):
                            
    # Create empty lists to store all data

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
    true_texts = []
    for _, row in df_human.iterrows():

        ehr_simulated = row['input_entities']
        entities.append(ehr_simulated)
        reports.append(row['true_text'])
        agents.append("human")
        sources.append(row['source'])
        specialties.append(row['specialty'])
        report_types.append(row['report_type'])
        true_texts.append(row['true_text'])
        
        accuracies.append('')
        fluencies.append('')
        compeltness.append('')

    for _, row in df_machine.iterrows():
        reports.append(row['generation'])
        ehr_simulated = row['input_entities']
        entities.append(ehr_simulated)

        agents.append("machine")
        sources.append(row['source'])
        specialties.append(row['specialty'])
        report_types.append(row['report_type'])
        true_texts.append(row['true_text'])

        accuracies.append('')
        fluencies.append('')
        compeltness.append('')

        
    df = pd.DataFrame({
        'EHR': entities,
        'true_texts': true_texts, 
        "Report": reports,
        "Agent": agents,
        "Source": sources,
        "Specialty": specialties,
        "Report type": report_types,
        'Accuracy': accuracies, 
        'Fluency': fluencies,
        'Completeness': compeltness,
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)


    df.to_excel(output_file, index=False)
    print(f"Excel file created successfully: {output_file}")
    

    return df


folder_path = '/Users/rouhizad/Documents/PycharmProjects/emerge-llm/code/Annotation/Journal/Phi4_outputs'
save_add = '/Users/rouhizad/Documents/PycharmProjects/emerge-llm/code/Annotation/Journal/Task2'
if not os.path.exists(save_add):
    os.mkdir(save_add)

csv_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.csv') and 'score' in file])
for cv_file in csv_files:

    df  = pd.read_csv(f'{folder_path}/{cv_file}', encoding='utf-8')
    df_human = df.sample(n=2) 
    df_machine = df.sample(n=15)
    lang = 'ENG' if 'ENG' in cv_file else 'FRE'
    save_add_name = f'{save_add}/Task2_case_reports_{lang}_' if 'PubMed' in cv_file else f'{save_add}/Task2_medical_transcripts_{lang}_'


    create_classification_csv(df_human, df_machine, save_add_name)

