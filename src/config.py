from pathlib import Path


# ============================================================
# PROJECT PATHS
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

RAW_DATA_FILE = RAW_DATA_DIR / "korean_migraine_study_translated.xlsx"

CLEAN_DATA_FILE = (
    PROCESSED_DATA_DIR / "migraine_daily_diary_clean.csv"
)

WEATHER_ENRICHED_DATA_FILE = (
    PROCESSED_DATA_DIR / "migraine_weather_enriched.csv"
)

FEATURE_ENGINEERED_DATA_FILE = (
    PROCESSED_DATA_DIR / "migraine_feature_engineered.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_STATE = 42


# ============================================================
# TARGET AND FILTERING VARIABLES
# ============================================================

TARGET = "Severity_VAS"

GROUP = "Registration No."

MIGRAINE_FILTER = (
    "Migraine (Y/N) Based on Premises from ICHD-3 Diagnosis"
)

SEVERITY_ORDER = [
    "Mild",
    "Moderate",
    "Severe",
]


# ============================================================
# PHYSIOLOGICAL FEATURES
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
# ENVIRONMENTAL DIARY FEATURES
# ============================================================

ENVIRONMENTAL_BINARY_FEATURES = [
    "External Factor: Excessive Sunlight",
    "External Factor: Noise",
    "External Factor: Inappropriate Lighting",
    "External Factor: Specific Smell (cosmetics, perfume, etc.)",
]

ENVIRONMENTAL_FEATURES = ENVIRONMENTAL_BINARY_FEATURES.copy()


# ============================================================
# LIFESTYLE AND BEHAVIORAL FEATURES
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
# TEMPORAL FEATURES
# ============================================================

# Start_Hour is excluded from modeling because it is represented
# cyclically by Start_Hour_sin and Start_Hour_cos.

TEMPORAL_FEATURES = [
    "Time_of_Day",
    "Start_Hour_sin",
    "Start_Hour_cos",
]


# ============================================================
# WEATHER FEATURES
# ============================================================

# ------------------------------------------------------------
# Raw weather features
# ------------------------------------------------------------

# surface_pressure is excluded because EDA showed that it was
# essentially redundant with pressure_msl.
#
# Raw wind_direction_10m is excluded from modeling because wind
# direction is represented cyclically using sine and cosine.

WEATHER_RAW_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
]


# ------------------------------------------------------------
# Cyclical weather features
# ------------------------------------------------------------

WEATHER_CYCLICAL_FEATURES = [
    "wind_direction_sin",
    "wind_direction_cos",
]


# ------------------------------------------------------------
# Signed weather-change features
# ------------------------------------------------------------

WEATHER_CHANGE_24H_FEATURES = [
    "relative_humidity_change_24h",
    "temperature_change_24h",
    "pressure_msl_change_24h",
    "wind_speed_10m_change_24h",
    "wind_direction_10m_change_24h",
]

WEATHER_CHANGE_48H_FEATURES = [
    "relative_humidity_change_48h",
    "temperature_change_48h",
    "pressure_msl_change_48h",
    "wind_speed_10m_change_48h",
    "wind_direction_10m_change_48h",
]

WEATHER_CHANGE_72H_FEATURES = [
    "relative_humidity_change_72h",
    "temperature_change_72h",
    "pressure_msl_change_72h",
    "wind_speed_10m_change_72h",
    "wind_direction_10m_change_72h",
]

WEATHER_CHANGE_FEATURES_ALL = (
    WEATHER_CHANGE_24H_FEATURES
    + WEATHER_CHANGE_48H_FEATURES
    + WEATHER_CHANGE_72H_FEATURES
)


# ------------------------------------------------------------
# Absolute weather-change features
# ------------------------------------------------------------

# These represent magnitude of weather change regardless of
# whether the original variable increased or decreased.

ABSOLUTE_WEATHER_CHANGE_FEATURES = [
    "abs_temperature_change_24h",
    "abs_relative_humidity_change_24h",
    "abs_pressure_msl_change_24h",
    "abs_wind_speed_10m_change_24h",

    "abs_temperature_change_48h",
    "abs_relative_humidity_change_48h",
    "abs_pressure_msl_change_48h",
    "abs_wind_speed_10m_change_48h",

    "abs_temperature_change_72h",
    "abs_relative_humidity_change_72h",
    "abs_pressure_msl_change_72h",
    "abs_wind_speed_10m_change_72h",
]


# ------------------------------------------------------------
# Complete weather feature set
# ------------------------------------------------------------

WEATHER_FEATURES = (
    WEATHER_RAW_FEATURES
    + WEATHER_CYCLICAL_FEATURES
    + WEATHER_CHANGE_FEATURES_ALL
    + ABSOLUTE_WEATHER_CHANGE_FEATURES
)


# ============================================================
# REUSABLE FEATURE GROUPS
# ============================================================

FEATURE_GROUPS = {
    "Physiological": PHYSIOLOGICAL_FEATURES,
    "Environmental": ENVIRONMENTAL_FEATURES,
    "Lifestyle": LIFESTYLE_FEATURES,
    "Temporal": TEMPORAL_FEATURES,
    "Weather": WEATHER_FEATURES,
}


# ============================================================
# FINAL MODELING FEATURE SETS
# ============================================================

# Physiological and lifestyle predictors

PHYSIOLOGICAL_LIFESTYLE_FEATURES = (
    PHYSIOLOGICAL_FEATURES
    + LIFESTYLE_FEATURES
)


# Environmental and temporal predictors

ENVIRONMENTAL_TEMPORAL_FEATURES = (
    ENVIRONMENTAL_FEATURES
    + WEATHER_FEATURES
    + TEMPORAL_FEATURES
)


# All candidate predictors

COMBINED_FEATURES = (
    PHYSIOLOGICAL_LIFESTYLE_FEATURES
    + ENVIRONMENTAL_TEMPORAL_FEATURES
)


# Feature sets used for domain comparison

FEATURE_SETS = {
    "Physiological + Lifestyle": PHYSIOLOGICAL_LIFESTYLE_FEATURES,
    "Environmental + Temporal": ENVIRONMENTAL_TEMPORAL_FEATURES,
    "Combined": COMBINED_FEATURES,
}


# ============================================================
# PLANNED MODELS AND METRICS
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