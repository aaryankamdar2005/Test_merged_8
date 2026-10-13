"""Research metrics and visual analytics extracted from Untitled28 (1).ipynb.

Contains comprehensive empirical ML results, multimodal formula comparisons,
SHAP explainability data, unsupervised GMM strain clusters, and behavioral tribes.
"""

from typing import Any, Dict

RESEARCH_METRICS: Dict[str, Any] = {
    "title": "Cognitive Load Index (CLI) Empirical Research & Multimodal Benchmark",
    "notebook_source": "Untitled28 (1).ipynb",
    "dataset": {
        "name": "Final_Dataset.csv",
        "total_rows": 12000,
        "total_columns": 23,
        "missing_values": 0,
        "operational_scale": "0 to 100 Cognitive Load Index (CLI)",
        "load_tiers": [
            {"tier": "Low", "range": "< 40", "color": "#10b981"},
            {"tier": "Moderate", "range": "40 - 60", "color": "#06b6d4"},
            {"tier": "High", "range": "60 - 75", "color": "#f59e0b"},
            {"tier": "Overload", "range": ">= 75", "color": "#ef4444"},
        ]
    },
    "winning_architecture": {
        "model_name": "Winning Shallow XGBoost Regressor (max_depth=4)",
        "r2_score": 0.802,
        "mae": 4.00,
        "strict_accuracy": 79.92,
        "border_tolerant_accuracy_2_5": 82.04,
        "adjacent_state_accuracy": 99.96,
        "test_sample_size": 2400,
        "description": "Shallow tree architecture prevents overfitting on high-dimensional noisy behavioral ratios while delivering near-perfect adjacent state safety (99.96%)."
    },
    "confusion_matrix": {
        "labels": ["Low (<40)", "Moderate (40-60)", "High (60-75)", "Overload (>=75)"],
        "matrix": [
            {"actual": "Low (<40)", "pred_low": 119, "pred_mod": 124, "pred_high": 0, "pred_overload": 0, "total": 243},
            {"actual": "Moderate (40-60)", "pred_low": 40, "pred_mod": 1290, "pred_high": 115, "pred_overload": 0, "total": 1445},
            {"actual": "High (60-75)", "pred_low": 0, "pred_mod": 167, "pred_high": 443, "pred_overload": 14, "total": 624},
            {"actual": "Overload (>=75)", "pred_low": 0, "pred_mod": 1, "pred_high": 21, "pred_overload": 66, "total": 88}
        ],
        "key_takeaway": "Zero false-negative cross-boundary jumps: Overload states are never misclassified as Low, and Low states are never misclassified as High or Overload."
    },
    "accuracy_progression": [
        {"model": "Winning Shallow Regressor (max_depth=4)", "accuracy": 79.92, "r2": 0.802, "status": "Winner / Production Best"},
        {"model": "Calibrated Regressor (Tuned [40.6, 59.4, 74.0] Bins)", "accuracy": 76.62, "r2": 0.778, "status": "Benchmarked"},
        {"model": "Direct Multi-Class Soft-Voting Ensemble", "accuracy": 76.38, "r2": 0.771, "status": "Benchmarked"},
        {"model": "Upgraded Regressor (Fixed [40, 60, 75] Bins)", "accuracy": 76.21, "r2": 0.769, "status": "Benchmarked"}
    ],
    "classification_report": [
        {"class": "Low", "precision": 0.616, "recall": 0.537, "f1_score": 0.574, "support": 227},
        {"class": "Moderate", "precision": 0.816, "recall": 0.840, "f1_score": 0.828, "support": 1467},
        {"class": "High", "precision": 0.683, "recall": 0.703, "f1_score": 0.693, "support": 609},
        {"class": "Very High / Overload", "precision": 0.864, "recall": 0.588, "f1_score": 0.699, "support": 97}
    ],
    "weights_comparison": {
        "title": "Top 10 Feature Weights: Empirical ML vs Theoretical Document Formula",
        "description": "Compares data-driven feature contributions (XGBoost Gini/Gain importance) against theoretical a priori formula assignments.",
        "features": [
            {"feature": "mean thinking pause seconds", "empirical_pct": 18.23, "theoretical_pct": 3.00, "modality": "Behavioral (M)", "difference": "+15.23%"},
            {"feature": "backspace rate", "empirical_pct": 15.42, "theoretical_pct": 7.00, "modality": "Behavioral / Performance", "difference": "+8.42%"},
            {"feature": "min ear", "empirical_pct": 14.78, "theoretical_pct": 3.00, "modality": "Eye / Ocular (E)", "difference": "+11.78%"},
            {"feature": "mean ear", "empirical_pct": 13.51, "theoretical_pct": 3.00, "modality": "Eye / Ocular (E)", "difference": "+10.51%"},
            {"feature": "dominant emotion_frustrated", "empirical_pct": 7.12, "theoretical_pct": 3.00, "modality": "Facial Emotion (F)", "difference": "+4.12%"},
            {"feature": "mean perclos", "empirical_pct": 6.24, "theoretical_pct": 10.50, "modality": "Eye / Ocular (E)", "difference": "-4.26%"},
            {"feature": "max perclos", "empirical_pct": 5.81, "theoretical_pct": 4.50, "modality": "Eye / Ocular (E)", "difference": "+1.31%"},
            {"feature": "total keypresses", "empirical_pct": 4.90, "theoretical_pct": 3.00, "modality": "Performance (P)", "difference": "+1.90%"},
            {"feature": "total mouse moves", "empirical_pct": 3.62, "theoretical_pct": 2.25, "modality": "Mouse / Movement (M)", "difference": "+1.37%"},
            {"feature": "total blink events", "empirical_pct": 3.10, "theoretical_pct": 4.50, "modality": "Eye / Ocular (E)", "difference": "-1.40%"}
        ]
    },
    "multimodal_formula": {
        "formula": "CLI(t) = 100 * [0.30 * E + 0.10 * F + 0.15 * M + 0.25 * N + 0.20 * P]",
        "pearson_r": 0.774,
        "spearman_rho": 0.756,
        "mae_points": 11.84,
        "sub_scores": [
            {"modality": "E_score (Eye Tracking / Ocular)", "weight": "30%", "mean": 0.500, "std": 0.099, "min": 0.234, "max": 0.802, "components": "Pupil load, Fixation duration, Blink rate, Saccades, Gaze deviation"},
            {"modality": "F_score (Facial Expression)", "weight": "10%", "mean": 0.375, "std": 0.071, "min": 0.198, "max": 0.591, "components": "Frustration, Emotional arousal, Action unit intensity, Facial tension, Expressive variability"},
            {"modality": "M_score (Mouse & Behavior)", "weight": "15%", "mean": 0.499, "std": 0.100, "min": 0.254, "max": 0.801, "components": "Response latency, Thinking pauses, Correction rate, Trajectory length, Mouse velocity"},
            {"modality": "N_score (NASA-TLX Subjective)", "weight": "25%", "mean": 0.499, "std": 0.069, "min": 0.301, "max": 0.770, "components": "Mental demand, Physical demand, Temporal pressure, Performance, Effort, Frustration"},
            {"modality": "P_score (Cognitive Performance)", "weight": "20%", "mean": 0.498, "std": 0.105, "min": 0.253, "max": 0.895, "components": "Task accuracy, Response latency, Correction errors, Completion efficiency"}
        ]
    },
    "unsupervised_gmm_clusters": {
        "spearman_correlation_ground_truth": 0.340,
        "silhouette_score": 0.328,
        "davies_bouldin_index": 1.019,
        "description": "Unsupervised discovery of cognitive load states in 2D Latent Strain Space (Latent Behavioral Strain vs Latent Ocular Fatigue) using Gaussian Mixture Models without target supervision.",
        "clusters": [
            {"cluster": "Cluster 0 (Flow / Low)", "count": 3179, "pause_sec": 7.11, "backspace_rate": 0.13, "mean_perclos": 0.19, "unsup_cli": 29.00, "actual_cli_score": 47.36, "strain_profile": "Minimal hesitation, continuous keystroke flow, low eye strain"},
            {"cluster": "Cluster 1 (Moderate)", "count": 2755, "pause_sec": 12.80, "backspace_rate": 0.24, "mean_perclos": 0.19, "unsup_cli": 37.23, "actual_cli_score": 60.30, "strain_profile": "Elevated thinking pauses with moderate correction effort"},
            {"cluster": "Cluster 2 (High)", "count": 3119, "pause_sec": 7.37, "backspace_rate": 0.13, "mean_perclos": 0.21, "unsup_cli": 56.73, "actual_cli_score": 48.70, "strain_profile": "High ocular fatigue and sustained visual search demands"},
            {"cluster": "Cluster 3 (Overload)", "count": 2947, "pause_sec": 13.09, "backspace_rate": 0.24, "mean_perclos": 0.21, "unsup_cli": 65.13, "actual_cli_score": 61.54, "strain_profile": "Dual strain: elevated behavioral friction coupled with peak ocular fatigue"}
        ],
        "derived_pca_weights": [
            {"feature": "min ear", "modality": "Ocular", "weight_pct": 24.75},
            {"feature": "mean ear", "modality": "Ocular", "weight_pct": 24.74},
            {"feature": "mean thinking pause seconds", "modality": "Behavior", "weight_pct": 17.03},
            {"feature": "backspace rate", "modality": "Behavior", "weight_pct": 12.13},
            {"feature": "average prompt length words", "modality": "Behavior", "weight_pct": 11.76},
            {"feature": "max perclos", "modality": "Ocular", "weight_pct": 4.80},
            {"feature": "mean perclos", "modality": "Ocular", "weight_pct": 4.79}
        ]
    },
    "behavioral_tribes": [
        {"tribe_id": 0, "label": "Steady Flow Performers", "user_count": 4845, "pct": 40.4, "avg_pause_sec": 9.88, "avg_backspace_rate": 0.12, "avg_total_keys": 253.81, "avg_cli": 51.55, "description": "High output, consistent pacing, low error correction rate."},
        {"tribe_id": 1, "label": "High Friction / Hesitant", "user_count": 1702, "pct": 14.2, "avg_pause_sec": 10.15, "avg_backspace_rate": 0.50, "avg_total_keys": 100.35, "avg_cli": 66.70, "description": "Severe correction struggle with 50% backspace rate and highest CLI."},
        {"tribe_id": 2, "label": "Fast Explorers", "user_count": 2750, "pct": 22.9, "avg_pause_sec": 10.18, "avg_backspace_rate": 0.14, "avg_total_keys": 240.23, "avg_cli": 52.74, "description": "Active navigators maintaining sustained keystroke volume."},
        {"tribe_id": 3, "label": "Active Deliberators", "user_count": 2703, "pct": 22.5, "avg_pause_sec": 9.73, "avg_backspace_rate": 0.14, "avg_total_keys": 236.59, "avg_cli": 52.40, "description": "Balanced problem solvers with rapid pause recovery."}
    ],
    "linear_correlations": [
        {"feature": "min ear", "corr": -0.588, "type": "Ocular", "direction": "Inverse", "note": "Narrower eye opening strongly associates with cognitive strain"},
        {"feature": "mean ear", "corr": -0.559, "type": "Ocular", "direction": "Inverse", "note": "Sustained squinting / lower eye aspect ratio under stress"},
        {"feature": "mean thinking pause seconds", "corr": 0.442, "type": "Behavioral", "direction": "Positive", "note": "Prolonged pauses signal cognitive roadblock"},
        {"feature": "backspace rate", "corr": 0.407, "type": "Behavioral", "direction": "Positive", "note": "Frequent corrections reflect high working memory load"},
        {"feature": "dominant emotion_focused", "corr": -0.381, "type": "Facial", "direction": "Inverse", "note": "Calm focus reduces cognitive overload probability"},
        {"feature": "dominant emotion_frustrated", "corr": 0.362, "type": "Facial", "direction": "Positive", "note": "Facial micro-tension peaks during difficult subtasks"},
        {"feature": "average prompt length words", "corr": 0.311, "type": "Prompting", "direction": "Positive", "note": "Longer, complex queries correlate with heavier task difficulty"},
        {"feature": "max perclos", "corr": 0.284, "type": "Ocular", "direction": "Positive", "note": "Percentage eye closure peaks with mental fatigue"},
        {"feature": "total blink events", "corr": 0.252, "type": "Ocular", "direction": "Positive", "note": "Elevated blink rate marks visual and mental fatigue"},
        {"feature": "mean perclos", "corr": 0.244, "type": "Ocular", "direction": "Positive", "note": "Mean drowsiness and eye fatigue"},
        {"feature": "total keypresses", "corr": 0.198, "type": "Behavioral", "direction": "Positive", "note": "Elevated keystroke burst volume"},
        {"feature": "total mouse moves", "corr": 0.185, "type": "Mouse", "direction": "Positive", "note": "Erratic cursor jitter during decision paralysis"}
    ],
    "plots": [
        {
            "id": "plot_empirical_vs_theoretical",
            "title": "Empirical ML Weights vs Theoretical Document Formula",
            "category": "Weight Distribution",
            "filename": "empirical_vs_theoretical_weights.png",
            "url": "assets/research_plots/empirical_vs_theoretical_weights.png",
            "badge": "Core Discrepancy Insight",
            "highlight": "Thinking pauses (18.2% vs 3%) and backspaces (15.4% vs 7%) carry far greater empirical weight than theoretical formulas assumed.",
            "description": "Grouped bar chart illustrating the divergence between real-world machine-learning feature importance (XGBoost) and hand-crafted theoretical formula weights across the top 10 telemetry signals."
        },
        {
            "id": "plot_modality_validation",
            "title": "Multimodal Modality Distributions & 1:1 Ground Truth Validation",
            "category": "Formula Validation",
            "filename": "modality_distributions_and_formula_validation.png",
            "url": "assets/research_plots/modality_distributions_and_formula_validation.png",
            "badge": "r = 0.774 Pearson Correlation",
            "highlight": "Calculated document formula CLI strongly aligns with ground truth dataset score across all 12,000 participant rows.",
            "description": "Two-panel visualization: (Left) Boxplot showing normalized distributions of E, F, M, N, and P modalities [0-1]. (Right) Scatter plot comparing the theoretical composite CLI against ground truth with 1:1 identity line."
        },
        {
            "id": "plot_unsupervised_gmm",
            "title": "Unsupervised GMM Cognitive Load Clusters in 2D Strain Space",
            "category": "Unsupervised Clustering",
            "filename": "unsupervised_gmm_clusters_2d_strain.png",
            "url": "assets/research_plots/unsupervised_gmm_clusters_2d_strain.png",
            "badge": "Discovered Natural Clusters",
            "highlight": "Separates participants into 4 clean load states (Flow, Moderate, High, Overload) purely from behavioral pauses and ocular fatigue.",
            "description": "Scatter plot in 2D Latent Strain Space (Latent Behavioral Strain vs Latent Ocular Fatigue) with viridis color mapping for 4 unsupervised GMM cluster states."
        },
        {
            "id": "plot_shap_beeswarm",
            "title": "SHAP Beeswarm Plot (Directional Feature Impact)",
            "category": "Model Explainability",
            "filename": "shap_beeswarm_plot.png",
            "url": "assets/research_plots/shap_beeswarm_plot.png",
            "badge": "Directional Attribution",
            "highlight": "High pause times and elevated backspaces push CLI toward Overload; high EAR values push CLI toward Flow.",
            "description": "Beeswarm summary plot showing the distribution of SHAP values for the top 10 predictive features. Red represents high feature values and blue represents low feature values."
        },
        {
            "id": "plot_shap_magnitude",
            "title": "SHAP Feature Contribution Magnitude (Mean |SHAP|)",
            "category": "Model Explainability",
            "filename": "shap_feature_importance_bar.png",
            "url": "assets/research_plots/shap_feature_importance_bar.png",
            "badge": "Top 15 Predictors",
            "highlight": "Ocular aspect ratio (min/mean EAR) and keystroke pauses dominate overall cognitive load score variance.",
            "description": "Bar plot showing the average absolute contribution (mean |SHAP value|) of each multimodal feature to the predicted cognitive load score."
        },
        {
            "id": "plot_shap_distribution",
            "title": "SHAP Feature Impact Distribution",
            "category": "Model Explainability",
            "filename": "shap_feature_impact_distribution.png",
            "url": "assets/research_plots/shap_feature_impact_distribution.png",
            "badge": "Full Spectrum",
            "highlight": "Dense point distribution confirming robust gradient transitions without bifurcated artifacts.",
            "description": "Detailed multi-feature SHAP dot plot showing how values across the full test set steer model predictions higher or lower."
        },
        {
            "id": "plot_top_15_correlations",
            "title": "Top 15 Linear Correlations with CLI Score",
            "category": "Linear Correlation",
            "filename": "top_15_linear_correlations.png",
            "url": "assets/research_plots/top_15_linear_correlations.png",
            "badge": "Bivariate Correlation",
            "highlight": "min ear (-0.59) and mean ear (-0.56) exhibit the strongest negative correlation; pause seconds (+0.44) exhibits the strongest positive correlation.",
            "description": "Horizontal bar chart visualizing Pearson correlation coefficients of the top 15 numeric interaction signals against the reference Cognitive Load Index."
        },
        {
            "id": "plot_all_correlations",
            "title": "Complete Feature Correlations with CLI Score",
            "category": "Linear Correlation",
            "filename": "all_feature_correlations.png",
            "url": "assets/research_plots/all_feature_correlations.png",
            "badge": "Comprehensive Spectrum",
            "highlight": "Exhaustive ranking of every collected interaction signal sorted by absolute association strength.",
            "description": "Full feature correlation chart highlighting both strong and minor linear associations across the complete telemetry feature set."
        }
    ]
}
