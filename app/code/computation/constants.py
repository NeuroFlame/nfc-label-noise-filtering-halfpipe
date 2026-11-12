"""Default computation parameter values."""

DEFAULT_SamplingThreshold = 0.7
DEFAULT_Iteration = 101
DEFAULT_NTree = 201
DEFAULT_LabelThreshold = 2
DEFAULT_TypicalThreshold = 0.8
DEFAULT_TruncationParameter = 0.2
DEFAULT_IgnoreSubjectsWithMissingData = False
DEFAULT_FNCDomainNames = []
DEFAULT_LogLevel = "info"

# Other Constants used
PATIENT_LABEL_VALUE = 1
CONTROL_LABEL_VALUE = 2
BOUNDARY_SUBJECT_LABEL_VALUE = (
    -1
)  # Subjects who do not cross the threshold after relabeling

PARAM_KEY_SUBJECT_ID = "subject_id"
PARAM_KEY_DIAGNOSIS = "diagnosis"
PARAM_KEY_PATIENT_LABEL = "PatientLabel"
PARAM_KEY_CONTROL_LABEL = "ControlLabel"
PARAM_KEY_FNC_DOMAIN_NAMES = "FNCDomainNames"


def get_label_definitions(patient_label, control_label):
    """Map the internal patient/control label values to their names and values."""
    label_definition = {
        str(PATIENT_LABEL_VALUE): {"name": patient_label, "value": PATIENT_LABEL_VALUE},
        str(CONTROL_LABEL_VALUE): {"name": control_label, "value": CONTROL_LABEL_VALUE},
    }

    return label_definition
