"""
Prompt templates for EHR simulation from case reports and transcripts.
Supports english and french languages.
"""

from typing import Tuple


# ============================================================================
# CASE REPORT PROMPTS - ENGLISH
# ============================================================================

def create_case_report_ehr_prompt_en(case_report: str) -> Tuple[str, str]:
    """Create EHR simulation prompt from case report (English)."""
    system = (
        'You are an expert in the medical field, specialising in analysing '
        'and writing case reports.'
    )
    
    user = f'''
**** Objective ****
As an experienced physician specializing in medical documentation and electronic health record (EHR) systems, your task is to simulate a realistic EHR entry based on the provided medical case report. Generate a synthetic EHR that captures all clinical information as it would have been originally recorded in the patient's electronic health record, either in structured fields or as free-text clinical notes.

*** Methodology: ***
1. Carefully read and analyze the entire case report
2. Identify all clinically relevant information and entities
3. Strictly maintain the chronological order of events as presented
4. Simulate the EHR with sufficient detail to enable faithful reconstruction of the original report
5. Ensure all simulated entries are medically accurate and precise
6. Include temporal relationships and clinical context

*** Format Requirements: ***
- Present the simulated EHR in a clear, structured format of your choice
- Maintain chronological accuracy
- Use precise medical terminology
- Include all relevant quantitative values and measurements
- Preserve causal and temporal relationships between entries
- If specific information is absent, do not speculate

**** Target Case Report ****

**Case Report:**
{case_report}

**Simulated EHR:**
'''
    
    return system, user


# ============================================================================
# CASE REPORT PROMPTS - FRENCH
# ============================================================================

def create_case_report_ehr_prompt_fr(case_report: str) -> Tuple[str, str]:
    """Create EHR simulation prompt from case report (French)."""
    system = (
        'Vous êtes un expert dans le domaine médical, spécialisé dans l\'analyse '
        'et la rédaction de rapports de cas.'
    )
    
    user = f'''
**** Objectif ****
En tant que médecin expérimenté spécialisé dans la documentation médicale et les systèmes de dossiers de santé électroniques (DSE), votre tâche consiste à simuler une entrée DSE réaliste basée sur le rapport de cas médical fourni. Générez un DSE synthétique qui capture toutes les informations cliniques telles qu'elles auraient été initialement enregistrées dans le dossier de santé électronique du patient, soit dans des champs structurés, soit sous forme de notes cliniques en texte libre.

*** Méthodologie: ***
1. Lire et analyser attentivement l'ensemble du rapport de cas
2. Identifier toutes les informations et entités cliniquement pertinentes
3. Maintenir strictement l'ordre chronologique des événements tels que présentés
4. Simuler le DSE avec suffisamment de détails pour permettre une reconstruction fidèle du rapport original
5. S'assurer que toutes les entrées simulées sont médicalement exactes et précises
6. Inclure les relations temporelles et le contexte clinique

*** Exigences de format: ***
- Présenter le DSE simulé dans un format clair et structuré de votre choix
- Maintenir l'exactitude chronologique
- Utiliser une terminologie médicale précise
- Inclure toutes les valeurs et mesures quantitatives pertinentes
- Préserver les relations causales et temporelles entre les entrées
- Si des informations spécifiques sont absentes, ne pas spéculer

**** Rapport de cas cible ****

**Rapport de cas:**
{case_report}

**DSE simulé:**
'''
    
    return system, user


# ============================================================================
# TRANSCRIPT PROMPTS - ENGLISH
# ============================================================================

def create_transcript_ehr_prompt_en(
    medical_report: str,
    specialty: str,
    report_type: str
) -> Tuple[str, str]:
    """Create EHR simulation prompt from medical transcript (English)."""
    system = (
        f'You are an expert in the medical field of {specialty}, specialising in '
        f'analysing and writing {report_type}s.'
    )
    
    user = f'''
**** Objective ****
As an experienced physician specializing in {specialty} and electronic health record (EHR) systems, your task is to identify and extract all relevant clinical entities from the provided {report_type}. These entities represent information that would have been initially recorded in the patient's electronic health record, either in structured fields or as free-text clinical notes. The extracted entities must contain sufficient detail to enable faithful reconstruction of the original report.

*** Methodology: ***
1. Carefully read and analyze the entire {report_type}
2. Identify all clinically relevant entities and information
3. Strictly maintain the chronological order of events as presented
4. Extract entities with sufficient detail to enable complete reconstruction of the original report
5. Ensure all extracted entities are medically accurate and precise
6. Include temporal relationships and clinical context between entities
7. Preserve causal and sequential connections between clinical events

*** Format Requirements: ***
- Present the extracted entities in a clear, structured format of your choice
- Maintain chronological accuracy throughout the extraction
- Use precise medical terminology
- Include all relevant quantitative values and measurements
- Preserve temporal and causal relationships between entities
- Pay particular attention to the sequence of events, as this is crucial for full understanding
- If specific information is absent, do not speculate or infer

**** Target Medical Report ****

**{report_type.capitalize()}:**
{medical_report}

**Simulated EHR:**
'''
    
    return system, user


# ============================================================================
# TRANSCRIPT PROMPTS - FRENCH
# ============================================================================

def create_transcript_ehr_prompt_fr(
    medical_report: str,
    specialty: str,
    report_type: str
) -> Tuple[str, str]:
    """Create EHR simulation prompt from medical transcript (French)."""
    system = (
        f'Vous êtes un expert dans le domaine médical de {specialty}, spécialisé dans '
        f'l\'analyse et la rédaction de {report_type}s.'
    )
    
    user = f'''
**** Objectif ****
En tant que médecin expérimenté spécialisé en {specialty} et dans les systèmes de dossiers de santé électroniques (DSE), votre tâche consiste à identifier et extraire toutes les entités cliniques pertinentes du {report_type} fourni. Ces entités représentent des informations qui auraient été initialement enregistrées dans le dossier de santé électronique du patient, soit dans des champs structurés, soit sous forme de notes cliniques en texte libre. Les entités extraites doivent contenir suffisamment de détails pour permettre une reconstruction fidèle du rapport original.

*** Méthodologie: ***
1. Lire et analyser attentivement l'ensemble du {report_type}
2. Identifier toutes les entités et informations cliniquement pertinentes
3. Maintenir strictement l'ordre chronologique des événements tels que présentés
4. Extraire les entités avec suffisamment de détails pour permettre une reconstruction complète du rapport original
5. S'assurer que toutes les entités extraites sont médicalement exactes et précises
6. Inclure les relations temporelles et le contexte clinique entre les entités
7. Préserver les connexions causales et séquentielles entre les événements cliniques

*** Exigences de format: ***
- Présenter les entités extraites dans un format clair et structuré de votre choix
- Maintenir l'exactitude chronologique tout au long de l'extraction
- Utiliser une terminologie médicale précise
- Inclure toutes les valeurs et mesures quantitatives pertinentes
- Préserver les relations temporelles et causales entre les entités
- Accorder une attention particulière à la séquence des événements, car cela est crucial pour une compréhension complète
- Si des informations spécifiques sont absentes, ne pas spéculer ou déduire

**** Rapport médical cible ****

**{report_type.capitalize()}:**
{medical_report}

**DSE simulé:**
'''
    
    return system, user


# ============================================================================
# PROMPT SELECTOR
# ============================================================================

def get_prompt_creator(task: str, language: str):
    """
    Get the appropriate prompt creator function.
    
    Args:
        task: 'case_report' or 'transcript'
        language: 'english' or 'french'
    
    Returns:
        Function to create prompts
    """
    if task == 'case_report':
        if language == 'french':
            return create_case_report_ehr_prompt_fr
        else:  # english
            return create_case_report_ehr_prompt_en
    
    elif task == 'transcript':
        if language == 'french':
            return create_transcript_ehr_prompt_fr
        else:  # english
            return create_transcript_ehr_prompt_en
