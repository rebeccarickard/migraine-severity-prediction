from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DATA_DIR = DATA_DIR / "Raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "Intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "Processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "Notebooks"
FIGURES_DIR = PROJECT_ROOT / "Figures and Tables"
RESULTS_DIR = PROJECT_ROOT / "Results"
MODELS_DIR = PROJECT_ROOT / "Models"

RAW_DATA_FILE = (RAW_DATA_DIR / "korean_migraine_study_translated.xlsx")

CLEAN_DATA_FILE = (PROCESSED_DATA_DIR / "migraine_daily_diary_clean.csv")


WEATHER_ENRICHED_DATA_FILE = PROCESSED_DATA_DIR / "migraine_weather_enriched.csv"

# ============================================================
# Reproducibility
# ============================================================

RANDOM_STATE = 42


# ============================================================
# Target and filtering variables
# ============================================================

TARGET = "Severity_VAS"

MIGRAINE_FILTER = (
    "Migraine (Y/N) Based on Premises from ICHD-3 Diagnosis"
)

SEVERITY_ORDER = [
    "Mild",
    "Moderate",
    "Severe",
]


# ============================================================
# Physiological features
# ============================================================

PHYSIOLOGICAL_BINARY_FEATURES = [
    "Preventive Medication Use",
    "Internal Factor: Stress",
    "Internal Factor: Oversleeping",
    "Internal Factor: Sleep Deprivation",
    "Internal Factor: Exercise",
    "Internal Factor: Lack of Exercise",
    "Internal Factor: Physical Fatigue",
    "Internal Factor: Menstrual Cycle: Menstruation",
    "Internal Factor: Menstrual Cycle: Ovulation",
    "Internal Factor: Excessive Emotional Change",
    "Other Factor: Irregular Meals (fasting, etc.)",
]

PHYSIOLOGICAL_ORDINAL_FEATURES = [
    "Total Exercise",
]

PHYSIOLOGICAL_FEATURES = (
    PHYSIOLOGICAL_BINARY_FEATURES
    + PHYSIOLOGICAL_ORDINAL_FEATURES
)


# ============================================================
# Environmental diary features
# ============================================================

ENVIRONMENTAL_BINARY_FEATURES = [
    "External Factor: Excessive Sunlight",
    "External Factor: Noise",
    "External Factor: Inappropriate Lighting",
    "External Factor: Specific Smell (cosmetics, perfume, etc.)",
]

ENVIRONMENTAL_FEATURES = ENVIRONMENTAL_BINARY_FEATURES.copy()


# ============================================================
# Lifestyle and behavioral features
# ============================================================

LIFESTYLE_BINARY_FEATURES = [
    "Other Factor: Excessive Alcohol",
    "Other Factor: Overeating",
    "Other Factor: Excessive Caffeine",
    "Other Factor: Excessive Smoking",
    "Other Factor: Cheese/Chocolate",
    "Other Factor: Travel",
    "Other Factor: Other",
]

LIFESTYLE_FEATURES = LIFESTYLE_BINARY_FEATURES.copy()


# ============================================================
# Temporal features
# ============================================================

TEMPORAL_FEATURES = [
    "Start_Hour",
    "Time_of_Day",
]


# ============================================================
# Weather API features
# ============================================================

WEATHER_RAW_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "pressure_msl",
]

WEATHER_CHANGE_24H_FEATURES = [
    "relative_humidity_change_24h",
    "temperature_change_24h",
    "pressure_msl_change_24h",
    "surface_pressure_change_24h",
]

WEATHER_CHANGE_48H_FEATURES = [
    "relative_humidity_change_48h",
    "temperature_change_48h",
    "pressure_msl_change_48h",
    "surface_pressure_change_48h",
]

WEATHER_CHANGE_72H_FEATURES = [
    "relative_humidity_change_72h",
    "temperature_change_72h",
    "pressure_msl_change_72h",
    "surface_pressure_change_72h",
]

WEATHER_CHANGE_FEATURES_ALL = (
    WEATHER_CHANGE_24H_FEATURES
    + WEATHER_CHANGE_48H_FEATURES
    + WEATHER_CHANGE_72H_FEATURES
)

WEATHER_FEATURES = (
    WEATHER_RAW_FEATURES
    + WEATHER_CHANGE_FEATURES_ALL
)


# ============================================================
# Reusable feature groups
# ============================================================

FEATURE_GROUPS = {
    "Physiological": PHYSIOLOGICAL_FEATURES,
    "Environmental": ENVIRONMENTAL_FEATURES,
    "Lifestyle": LIFESTYLE_FEATURES,
    "Temporal": TEMPORAL_FEATURES,
    "Weather": WEATHER_FEATURES,
}


# ============================================================
# Feature sets for model comparisons
# ============================================================

FEATURE_SETS = {
    "Physiological Only": PHYSIOLOGICAL_FEATURES,

    "Environmental Only": (
        ENVIRONMENTAL_FEATURES
        + TEMPORAL_FEATURES
        + WEATHER_FEATURES
    ),

    "Combined": (
        PHYSIOLOGICAL_FEATURES
        + ENVIRONMENTAL_FEATURES
        + TEMPORAL_FEATURES
        + WEATHER_FEATURES
    ),

    "All Features": (
        PHYSIOLOGICAL_FEATURES
        + ENVIRONMENTAL_FEATURES
        + LIFESTYLE_FEATURES
        + TEMPORAL_FEATURES
        + WEATHER_FEATURES
    ),
}


# ============================================================
# Planned models and metrics
# ============================================================

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
]

EVALUATION_METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro F1",
    "Weighted F1",
    "Per-Class Precision",
    "Per-Class Recall",
    "Per-Class F1",
    "Confusion Matrix",
]