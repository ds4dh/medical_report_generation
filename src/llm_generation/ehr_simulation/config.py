"""
Configuration for EHR simulation (case reports & transcripts).
"""

# Task types
SUPPORTED_TASKS = ['case_report', 'transcript']
SUPPORTED_LANGUAGES = ['english', 'french']

# Default model settings
DEFAULT_MODEL = 'microsoft/phi-4'
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_MODEL_LEN = 16000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.95

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.85
DEFAULT_MIN_P = 0.0
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 10000

# CSV column requirements by task
TASK_COLUMNS = {
    'case_report': {
        'required': ['id', 'language', 'case_report'],
        'output': ['id', 'language', 'case_report', 'simulated_ehr']
    },
    'transcript': {
        'required': ['id', 'eng_report', 'fre_report', 'eng_specialty', 'fre_specialty', 
                    'eng_report_type', 'fre_report_type'],
        'output': ['id', 'medical_report', 'specialty', 'report_type', 'simulated_ehr']
    }
}
