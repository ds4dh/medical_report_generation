"""
Configuration for medical text generation (case reports & transcripts).
"""

# Task types
SUPPORTED_TASKS = ['case_report', 'transcript']
SUPPORTED_LANGUAGES = ['english', 'french']
SUPPORTED_APPROACHES = ['zeroshot', 'fewshot']

# Default model settings
DEFAULT_MODEL = 'microsoft/phi-4'
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_SHOTS = 3
DEFAULT_MAX_MODEL_LEN = 16000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.85
DEFAULT_MIN_P = 0.0
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 12000

# CSV column requirements by task
TASK_COLUMNS = {
    'case_report': {
        'required': ['ehr', 'case_report', 'language'],
        'output': ['id', 'language', 'ehr', 'reference_report', 'generated_report']
    },
    'transcript': {
        'required': ['eng_ehr', 'fre_ehr', 'eng_report', 'fre_report', 
                    'eng_specialty', 'fre_specialty', 'eng_report_type', 'fre_report_type'],
        'output': ['ehr', 'reference_report', 'generated_report', 
                  'doctor_specialty', 'report_type']
    }
}
