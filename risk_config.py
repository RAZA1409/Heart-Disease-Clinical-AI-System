"""
Centralized Clinical Risk Classification Configuration.
Standardized thresholds across backend, CLI, dashboard, and UI templates.
"""

# Thresholds as percentages (0 - 100)
LOW_RISK_THRESHOLD = 30.0        # Probability < 30% -> LOW
MODERATE_RISK_THRESHOLD = 70.0   # 30% <= Probability < 70% -> MODERATE
                                 # Probability >= 70% -> HIGH

def classify_risk(probability_pct: float) -> str:
    """
    Classify risk level based on probability percentage (0.0 to 100.0).
    """
    try:
        prob = float(probability_pct)
    except (TypeError, ValueError):
        prob = 0.0

    if prob < LOW_RISK_THRESHOLD:
        return "LOW"
    elif prob < MODERATE_RISK_THRESHOLD:
        return "MODERATE"
    else:
        return "HIGH"

def get_risk_color(risk_level: str) -> str:
    """
    Standardized HEX colors for risk levels.
    """
    colors = {
        "LOW": "#22c55e",       # Green
        "MODERATE": "#f59e0b",  # Orange
        "HIGH": "#ef4444"       # Red
    }
    return colors.get(str(risk_level).upper(), "#6b7280")

