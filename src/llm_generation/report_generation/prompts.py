"""
Prompt templates for case reports and medical transcripts.
Supports english and french languages.
"""

from typing import Tuple, List


# ============================================================================
# CASE REPORT PROMPTS - ENGLISH
# ============================================================================

def get_case_report_system_en() -> str:
    """System prompt for case reports (English)."""
    return (
        'You are an expert in the medical field, specialising in analysing '
        'and writing case reports.'
    )


def get_case_report_instructions_en() -> str:
    """Instructions for case reports (English)."""
    return """**** Objective ****
As an experienced physician who specializes in medical documentation, your task is to produce a detailed and structured clinical case report in English based on the provided Electronic Health Record (EHR).

**** Methodology to follow: ****
1. Carefully analyze all EHR data thoroughly
2. Strictly adhere to the chronological order of clinical events
3. Establish logical connections between symptoms, diagnoses, and treatments
4. Incorporate all relevant information without omissions
5. Use precise and appropriate medical terminology
6. Maintain an objective, factual, and professional tone

**** Format Requirements ****
- A single coherent and fluid paragraph
- Concise yet comprehensive writing style (following PubMed case report conventions)
- Technical vocabulary conforming to medical standards
- Sentence structures typical of published case reports
- If information is missing, omit rather than speculate"""


def create_case_report_fewshot_examples_en(dev_ehrs: List[str], dev_reports: List[str], num_shots: int) -> str:
    """Create few-shot examples for case reports (English)."""
    num_shots = min(num_shots, len(dev_ehrs))
    examples = []
    for idx in range(num_shots):
        examples.append(f"""Sample ({idx + 1}):
**Input EHR:**
{dev_ehrs[idx]}

**Generated Case Report:**
{dev_reports[idx]}

""")
    return '\n'.join(examples)


def create_case_report_prompt_en(input_ehr: str, fewshot_examples: str = None) -> Tuple[str, str]:
    """Create prompt for case reports (English)."""
    system = get_case_report_system_en()
    instructions = get_case_report_instructions_en()
    
    if fewshot_examples:
        user = f"""{instructions}

**** Few-Shot Examples ****
{fewshot_examples}

**** Target Case ****

**Input EHR:**
{input_ehr}

**Generated Case Report:**
"""
    else:
        user = f"""{instructions}

**** Target Case ****

**Input EHR:**
{input_ehr}

**Generated Case Report:**
"""
    
    return system, user


# ============================================================================
# CASE REPORT PROMPTS - FRENCH
# ============================================================================

def get_case_report_system_fr() -> str:
    """System prompt for case reports (French)."""
    return (
        'Vous êtes un expert dans le domaine médical, spécialisé dans l\'analyse '
        'et la rédaction de rapports de cas.'
    )


def get_case_report_instructions_fr() -> str:
    """Instructions for case reports (French)."""
    return """**** Objectif ****
En tant que médecin expérimenté spécialisé dans la documentation médicale, votre tâche consiste à produire un rapport de cas clinique détaillé et structuré en français basé sur le Dossier de Santé Électronique (DSE) fourni.

**** Méthodologie à suivre: ****
1. Analyser soigneusement toutes les données du DSE de manière approfondie
2. Respecter strictement l'ordre chronologique des événements cliniques
3. Établir des connexions logiques entre les symptômes, les diagnostics et les traitements
4. Incorporer toutes les informations pertinentes sans omissions
5. Utiliser une terminologie médicale précise et appropriée
6. Maintenir un ton objectif, factuel et professionnel

**** Exigences de format ****
- Un paragraphe unique cohérent et fluide
- Style d'écriture concis mais complet (suivant les conventions des rapports de cas PubMed)
- Vocabulaire technique conforme aux normes médicales
- Structures de phrases typiques des rapports de cas publiés
- Si des informations manquent, les omettre plutôt que de spéculer"""


def create_case_report_fewshot_examples_fr(dev_ehrs: List[str], dev_reports: List[str], num_shots: int) -> str:
    """Create few-shot examples for case reports (French)."""
    num_shots = min(num_shots, len(dev_ehrs))
    examples = []
    for idx in range(num_shots):
        examples.append(f"""Exemple ({idx + 1}):
**DSE en entrée:**
{dev_ehrs[idx]}

**Rapport de cas généré:**
{dev_reports[idx]}

""")
    return '\n'.join(examples)


def create_case_report_prompt_fr(input_ehr: str, fewshot_examples: str = None) -> Tuple[str, str]:
    """Create prompt for case reports (French)."""
    system = get_case_report_system_fr()
    instructions = get_case_report_instructions_fr()
    
    if fewshot_examples:
        user = f"""{instructions}

**** Exemples Few-Shot ****
{fewshot_examples}

**** Cas cible ****

**DSE en entrée:**
{input_ehr}

**Rapport de cas généré:**
"""
    else:
        user = f"""{instructions}

**** Cas cible ****

**DSE en entrée:**
{input_ehr}

**Rapport de cas généré:**
"""
    
    return system, user


# ============================================================================
# TRANSCRIPT PROMPTS - ENGLISH
# ============================================================================

def create_transcript_fewshot_examples_en(
    dev_ehrs: List[str], 
    dev_reports: List[str],
    dev_specialties: List[str],
    dev_report_types: List[str],
    target_specialty: str,
    target_report_type: str,
    num_shots: int
) -> str:
    """Create few-shot examples for transcripts matching specialty/type (English)."""
    examples = []
    count = 0
    
    for idx in range(len(dev_ehrs)):
        if count == num_shots:
            break
        if dev_specialties[idx] == target_specialty and dev_report_types[idx] == target_report_type:
            examples.append(f"""Sample ({count + 1}):
** Input EHR: **
{dev_ehrs[idx]}

** Generated Medical Report: **
{dev_reports[idx]}

""")
            count += 1
    
    return '\n'.join(examples)


def create_transcript_prompt_en(
    input_ehr: str,
    specialty: str,
    report_type: str,
    fewshot_examples: str = None
) -> Tuple[str, str]:
    """Create prompt for transcripts (English)."""
    system = f'As an experienced {specialty} physician, you are tasked with producing a detailed and structured {report_type} based solely on the provided medical information.'
    
    if fewshot_examples:
        user = f"""
Your task is to write a {report_type} medical transcript based **only** on the provided EHR data.

Requirements:
- The report must be detailed, clear, and written in a formal professional style typical of {report_type} transcripts.
- All provided information must be incorporated accurately into the report. Do not invent new details outside the given data.
- Use the structure and tone illustrated in the few-shot examples.
- If any information is not included in the EHR, do not include relevant section in the report

**** Few-Shot Examples ****
{fewshot_examples}

**** Target Case ****
** Input EHR **
{input_ehr}

** Generated Medical Report **
"""
    else:
        user = f"""***** Objective: Create a professional {report_type} as a medical specialist in {specialty} *****

As an experienced {specialty} physician, you are tasked with producing a detailed and structured {report_type} based solely on the provided medical information.

***** Methodology to follow:
1. Carefully analyze each provided medical information
2. Strictly adhere to the chronological order of clinical events
3. Establish logical connections between symptoms, diagnoses, and treatments
4. Incorporate all relevant information without omissions
5. Use precise and appropriate medical terminology
6. Maintain an objective, factual, and professional tone
7. Organize content according to standard clinical presentation format:
    - Begin with patient demographics and chief complaint
    - Follow with relevant history and presentation
    - Include examination findings and diagnostic results
    - Detail diagnoses in order of clinical priority
    - Conclude with treatments, interventions, and recommendations
8. Write in the formal, professional style typical of {report_type} medical documentation

***** Instructions for your {report_type}:
- Create a {report_type} that would be recognized as authentic by other {specialty} specialists
- Follow standard documentation conventions for this type of medical report
- Include all sections typically found in a {report_type}
- Concise yet comprehensive writing style

Produce a document that could be used in a real medical record by a healthcare professional.

***** Input EHR data for report generation:
{input_ehr}

***** Output {report_type}: 
"""
    
    return system, user


# ============================================================================
# TRANSCRIPT PROMPTS - FRENCH
# ============================================================================

def create_transcript_fewshot_examples_fr(
    dev_ehrs: List[str], 
    dev_reports: List[str],
    dev_specialties: List[str],
    dev_report_types: List[str],
    target_specialty: str,
    target_report_type: str,
    num_shots: int
) -> str:
    """Create few-shot examples for transcripts matching specialty/type (French)."""
    examples = []
    count = 0
    
    for idx in range(len(dev_ehrs)):
        if count == num_shots:
            break
        if dev_specialties[idx] == target_specialty and dev_report_types[idx] == target_report_type:
            examples.append(f"""Exemple ({count + 1}):
** DSE en entrée: **
{dev_ehrs[idx]}

** Rapport médical généré: **
{dev_reports[idx]}

""")
            count += 1
    
    return '\n'.join(examples)


def create_transcript_prompt_fr(
    input_ehr: str,
    specialty: str,
    report_type: str,
    fewshot_examples: str = None
) -> Tuple[str, str]:
    """Create prompt for transcripts (French)."""
    system = f'En tant que médecin expérimenté en {specialty}, vous êtes chargé de produire un {report_type} détaillé et structuré basé uniquement sur les informations médicales fournies.'
    
    if fewshot_examples:
        user = f"""
Votre tâche consiste à rédiger une transcription médicale de {report_type} basée **uniquement** sur les données DSE fournies.

Exigences:
- Le rapport doit être détaillé, clair et rédigé dans un style professionnel formel typique des transcriptions de {report_type}.
- Toutes les informations fournies doivent être incorporées avec précision dans le rapport. N'inventez pas de nouveaux détails en dehors des données données.
- Utilisez la structure et le ton illustrés dans les exemples few-shot.
- Si des informations ne sont pas incluses dans le DSE, n'incluez pas la section pertinente dans le rapport

**** Exemples Few-Shot ****
{fewshot_examples}

**** Cas cible ****
** DSE en entrée **
{input_ehr}

** Rapport médical généré **
"""
    else:
        user = f"""***** Objectif: Créer un {report_type} professionnel en tant que spécialiste médical en {specialty} *****

En tant que médecin expérimenté en {specialty}, vous êtes chargé de produire un {report_type} détaillé et structuré basé uniquement sur les informations médicales fournies.

***** Méthodologie à suivre:
1. Analyser attentivement chaque information médicale fournie
2. Respecter strictement l'ordre chronologique des événements cliniques
3. Établir des connexions logiques entre les symptômes, les diagnostics et les traitements
4. Incorporer toutes les informations pertinentes sans omissions
5. Utiliser une terminologie médicale précise et appropriée
6. Maintenir un ton objectif, factuel et professionnel
7. Organiser le contenu selon le format de présentation clinique standard:
    - Commencer par les données démographiques du patient et le motif de consultation
    - Suivre avec l'historique pertinent et la présentation
    - Inclure les résultats de l'examen et les résultats diagnostiques
    - Détailler les diagnostics par ordre de priorité clinique
    - Conclure avec les traitements, les interventions et les recommandations
8. Rédiger dans le style formel et professionnel typique de la documentation médicale {report_type}

***** Instructions pour votre {report_type}:
- Créer un {report_type} qui serait reconnu comme authentique par d'autres spécialistes en {specialty}
- Suivre les conventions de documentation standard pour ce type de rapport médical
- Inclure toutes les sections généralement trouvées dans un {report_type}
- Style d'écriture concis mais complet

Produire un document qui pourrait être utilisé dans un dossier médical réel par un professionnel de la santé.

***** Données DSE en entrée pour la génération du rapport:
{input_ehr}

***** Sortie {report_type}: 
"""
    
    return system, user


# ============================================================================
# PROMPT SELECTOR
# ============================================================================

def get_prompt_creator(task: str, language: str):
    """
    Get the appropriate prompt creator functions.
    
    Args:
        task: 'case_report' or 'transcript'
        language: 'english' or 'french'
    """
    if task == 'case_report':
        if language == 'french':
            return {
                'create_fewshot_examples': create_case_report_fewshot_examples_fr,
                'create_prompt': create_case_report_prompt_fr,
            }
        else:  # english
            return {
                'create_fewshot_examples': create_case_report_fewshot_examples_en,
                'create_prompt': create_case_report_prompt_en,
            }
    
    elif task == 'transcript':
        if language == 'french':
            return {
                'create_fewshot_examples': create_transcript_fewshot_examples_fr,
                'create_prompt': create_transcript_prompt_fr,
            }
        else:  # english
            return {
                'create_fewshot_examples': create_transcript_fewshot_examples_en,
                'create_prompt': create_transcript_prompt_en,
            }
