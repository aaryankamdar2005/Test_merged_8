/**
 * Research Metrics & Model Visualizations Viewer
 * Generated from Untitled28 (1).ipynb
 * Displays empirical ML weights, multimodal formula validation,
 * SHAP explainability, unsupervised GMM strain space, and behavioral tribes.
 */

export const RESEARCH_METRICS_DATA = {
  title: "Cognitive Load Index (CLI) Empirical Research & Multimodal Benchmark",
  notebook_source: "Untitled28 (1).ipynb",
  dataset: {
    name: "Final_Dataset.csv",
    total_rows: 12000,
    total_columns: 23,
    missing_values: 0,
    operational_scale: "0 to 100 Cognitive Load Index (CLI)",
    load_tiers: [
      { tier: "Low", range: "< 40", color: "#10b981" },
      { tier: "Moderate", range: "40 - 60", color: "#06b6d4" },
      { tier: "High", range: "60 - 75", color: "#f59e0b" },
      { tier: "Overload", range: ">= 75", color: "#ef4444" }
    ]
  },
  winning_architecture: {
    model_name: "Winning Shallow XGBoost Regressor (max_depth=4)",
    r2_score: 0.802,
    mae: 4.00,
    strict_accuracy: 79.92,
    border_tolerant_accuracy_2_5: 82.04,
    adjacent_state_accuracy: 99.96,
    test_sample_size: 2400,
    description: "Shallow tree architecture prevents overfitting on high-dimensional noisy behavioral ratios while delivering near-perfect adjacent state safety (99.96%)."
  },
  confusion_matrix: {
    labels: ["Low (<40)", "Moderate (40-60)", "High (60-75)", "Overload (>=75)"],
    matrix: [
      { actual: "Low (<40)", pred_low: 119, pred_mod: 124, pred_high: 0, pred_overload: 0, total: 243 },
      { actual: "Moderate (40-60)", pred_low: 40, pred_mod: 1290, pred_high: 115, pred_overload: 0, total: 1445 },
      { actual: "High (60-75)", pred_low: 0, pred_mod: 167, pred_high: 443, pred_overload: 14, total: 624 },
      { actual: "Overload (>=75)", pred_low: 0, pred_mod: 1, pred_high: 21, pred_overload: 66, total: 88 }
    ],
    key_takeaway: "Zero false-negative cross-boundary jumps: Overload states are never misclassified as Low, and Low states are never misclassified as High or Overload."
  },
  accuracy_progression: [
    { model: "Winning Shallow Regressor (max_depth=4)", accuracy: 79.92, r2: 0.802, status: "Winner (Production)" },
    { model: "Calibrated Regressor (Tuned [40.6, 59.4, 74.0] Bins)", accuracy: 76.62, r2: 0.778, status: "Benchmarked" },
    { model: "Direct Multi-Class Soft-Voting Ensemble", accuracy: 76.38, r2: 0.771, status: "Benchmarked" },
    { model: "Upgraded Regressor (Fixed [40, 60, 75] Bins)", accuracy: 76.21, r2: 0.769, status: "Benchmarked" }
  ],
  classification_report: [
    { class: "Low (<40)", precision: 0.616, recall: 0.537, f1_score: 0.574, support: 227 },
    { class: "Moderate (40-60)", precision: 0.816, recall: 0.840, f1_score: 0.828, support: 1467 },
    { class: "High (60-75)", precision: 0.683, recall: 0.703, f1_score: 0.693, support: 609 },
    { class: "Very High / Overload (≥75)", precision: 0.864, recall: 0.588, f1_score: 0.699, support: 97 }
  ],
  weights_comparison: {
    title: "Top 10 Feature Weights: Empirical ML vs Theoretical Document Formula",
    description: "Compares real data XGBoost feature importances (%) against theoretical formula weights assigned a priori.",
    features: [
      { name: "Thinking Pause Duration", code: "mean thinking pause seconds", empirical_pct: 18.23, theoretical_pct: 3.00, modality: "Behavioral", difference: "+15.23%" },
      { name: "Backspace Correction Rate", code: "backspace rate", empirical_pct: 15.42, theoretical_pct: 7.00, modality: "Behavior / Performance", difference: "+8.42%" },
      { name: "Min Eye Aspect Ratio", code: "min ear", empirical_pct: 14.78, theoretical_pct: 3.00, modality: "Eye / Ocular", difference: "+11.78%" },
      { name: "Mean Eye Aspect Ratio", code: "mean ear", empirical_pct: 13.51, theoretical_pct: 3.00, modality: "Eye / Ocular", difference: "+10.51%" },
      { name: "Dominant Frustration", code: "dominant emotion_frustrated", empirical_pct: 7.12, theoretical_pct: 3.00, modality: "Facial Emotion", difference: "+4.12%" },
      { name: "Mean PERCLOS (Eye Closure)", code: "mean perclos", empirical_pct: 6.24, theoretical_pct: 10.50, modality: "Eye / Ocular", difference: "-4.26%" },
      { name: "Max PERCLOS (Peak Closure)", code: "max perclos", empirical_pct: 5.81, theoretical_pct: 4.50, modality: "Eye / Ocular", difference: "+1.31%" },
      { name: "Total Keystrokes Volume", code: "total keypresses", empirical_pct: 4.90, theoretical_pct: 3.00, modality: "Task Performance", difference: "+1.90%" },
      { name: "Cursor Navigation Moves", code: "total mouse moves", empirical_pct: 3.62, theoretical_pct: 2.25, modality: "Mouse Tracking", difference: "+1.37%" },
      { name: "Total Blink Events", code: "total blink events", empirical_pct: 3.10, theoretical_pct: 4.50, modality: "Eye / Ocular", difference: "-1.40%" }
    ]
  },
  multimodal_formula: {
    formula: "CLI(t) = 100 × [ 0.30·E + 0.10·F + 0.15·M + 0.25·N + 0.20·P ]",
    pearson_r: 0.774,
    spearman_rho: 0.756,
    mae_points: 11.84,
    sub_scores: [
      { modality: "E_score (Eye / Ocular)", weight: "30%", mean: 0.500, std: 0.099, min: 0.234, max: 0.802, components: "Pupil load, Fixation duration, Blink rate, Saccades, Gaze deviation" },
      { modality: "F_score (Facial Expression)", weight: "10%", mean: 0.375, std: 0.071, min: 0.198, max: 0.591, components: "Frustration, Emotional arousal, Action unit intensity, Facial tension, Variability" },
      { modality: "M_score (Mouse & Behavior)", weight: "15%", mean: 0.499, std: 0.100, min: 0.254, max: 0.801, components: "Response latency, Thinking pauses, Correction rate, Trajectory length, Velocity" },
      { modality: "N_score (NASA-TLX Subjective)", weight: "25%", mean: 0.499, std: 0.069, min: 0.301, max: 0.770, components: "Mental demand, Physical demand, Temporal pressure, Performance, Effort, Frustration" },
      { modality: "P_score (Cognitive Performance)", weight: "20%", mean: 0.498, std: 0.105, min: 0.253, max: 0.895, components: "Task accuracy, Response latency, Correction errors, Completion efficiency" }
    ]
  },
  unsupervised_gmm_clusters: {
    spearman_correlation_ground_truth: 0.340,
    silhouette_score: 0.328,
    davies_bouldin_index: 1.019,
    description: "Unsupervised discovery of cognitive load states in 2D Latent Strain Space (Latent Behavioral Strain vs Latent Ocular Fatigue) using Gaussian Mixture Models without target supervision.",
    clusters: [
      { cluster: "Cluster 0 (Flow / Low)", count: 3179, pause_sec: 7.11, backspace_rate: 0.13, mean_perclos: 0.19, unsup_cli: 29.00, actual_cli_score: 47.36, strain_profile: "Minimal hesitation, continuous keystroke flow, low eye strain" },
      { cluster: "Cluster 1 (Moderate)", count: 2755, pause_sec: 12.80, backspace_rate: 0.24, mean_perclos: 0.19, unsup_cli: 37.23, actual_cli_score: 60.30, strain_profile: "Elevated thinking pauses with moderate correction effort" },
      { cluster: "Cluster 2 (High)", count: 3119, pause_sec: 7.37, backspace_rate: 0.13, mean_perclos: 0.21, unsup_cli: 56.73, actual_cli_score: 48.70, strain_profile: "High ocular fatigue and sustained visual search demands" },
      { cluster: "Cluster 3 (Overload)", count: 2947, pause_sec: 13.09, backspace_rate: 0.24, mean_perclos: 0.21, unsup_cli: 65.13, actual_cli_score: 61.54, strain_profile: "Dual strain: elevated behavioral friction coupled with peak ocular fatigue" }
    ],
    derived_pca_weights: [
      { name: "Min Eye Aspect Ratio", code: "min ear", modality: "Ocular", weight_pct: 24.75 },
      { name: "Mean Eye Aspect Ratio", code: "mean ear", modality: "Ocular", weight_pct: 24.74 },
      { name: "Thinking Pause Duration", code: "mean thinking pause seconds", modality: "Behavior", weight_pct: 17.03 },
      { name: "Backspace Correction Rate", code: "backspace rate", modality: "Behavior", weight_pct: 12.13 },
      { name: "Average Prompt Length", code: "average prompt length words", modality: "Behavior", weight_pct: 11.76 },
      { name: "Max PERCLOS (Peak Closure)", code: "max perclos", modality: "Ocular", weight_pct: 4.80 },
      { name: "Mean PERCLOS (Eye Closure)", code: "mean perclos", modality: "Ocular", weight_pct: 4.79 }
    ]
  },
  behavioral_tribes: [
    { tribe_id: 0, label: "Steady Flow Performers", user_count: 4845, pct: 40.4, avg_pause_sec: 9.88, avg_backspace_rate: 0.12, avg_total_keys: 253.81, avg_cli: 51.55, description: "High output, consistent pacing, low error correction rate." },
    { tribe_id: 1, label: "High Friction / Hesitant", user_count: 1702, pct: 14.2, avg_pause_sec: 10.15, avg_backspace_rate: 0.50, avg_total_keys: 100.35, avg_cli: 66.70, description: "Severe correction struggle with 50% backspace rate and highest CLI." },
    { tribe_id: 2, label: "Fast Explorers", user_count: 2750, pct: 22.9, avg_pause_sec: 10.18, avg_backspace_rate: 0.14, avg_total_keys: 240.23, avg_cli: 52.74, description: "Active navigators maintaining sustained keystroke volume." },
    { tribe_id: 3, label: "Active Deliberators", user_count: 2703, pct: 22.5, avg_pause_sec: 9.73, avg_backspace_rate: 0.14, avg_total_keys: 236.59, avg_cli: 52.40, description: "Balanced problem solvers with rapid pause recovery." }
  ],
  linear_correlations: [
    { name: "Min Eye Aspect Ratio", code: "min ear", corr: -0.588, type: "Ocular", direction: "Inverse", note: "Narrower eye opening strongly associates with cognitive strain" },
    { name: "Mean Eye Aspect Ratio", code: "mean ear", corr: -0.559, type: "Ocular", direction: "Inverse", note: "Sustained squinting / lower eye aspect ratio under stress" },
    { name: "Thinking Pause Duration", code: "mean thinking pause seconds", corr: 0.442, type: "Behavioral", direction: "Positive", note: "Prolonged pauses signal cognitive roadblock" },
    { name: "Backspace Correction Rate", code: "backspace rate", corr: 0.407, type: "Behavioral", direction: "Positive", note: "Frequent corrections reflect high working memory load" },
    { name: "Dominant Emotion: Focused", code: "dominant emotion_focused", corr: -0.381, type: "Facial", direction: "Inverse", note: "Calm focus reduces cognitive overload probability" },
    { name: "Dominant Emotion: Frustrated", code: "dominant emotion_frustrated", corr: 0.362, type: "Facial", direction: "Positive", note: "Facial micro-tension peaks during difficult subtasks" },
    { name: "Average Prompt Length", code: "average prompt length words", corr: 0.311, type: "Prompting", direction: "Positive", note: "Longer, complex queries correlate with heavier task difficulty" },
    { name: "Max PERCLOS (Peak Closure)", code: "max perclos", corr: 0.284, type: "Ocular", direction: "Positive", note: "Percentage eye closure peaks with mental fatigue" },
    { name: "Total Blink Events", code: "total blink events", corr: 0.252, type: "Ocular", direction: "Positive", note: "Elevated blink rate marks visual and mental fatigue" },
    { name: "Mean PERCLOS (Eye Closure)", code: "mean perclos", corr: 0.244, type: "Ocular", direction: "Positive", note: "Mean drowsiness and eye fatigue" },
    { name: "Total Keystrokes Volume", code: "total keypresses", corr: 0.198, type: "Behavioral", direction: "Positive", note: "Elevated keystroke burst volume" },
    { name: "Total Mouse Moves", code: "total mouse moves", corr: 0.185, type: "Mouse", direction: "Positive", note: "Erratic cursor jitter during decision paralysis" }
  ],
  plots: [
    {
      id: "plot_empirical_vs_theoretical",
      title: "Empirical ML Weights vs Theoretical Document Formula",
      category: "Weight Discrepancy",
      filename: "empirical_vs_theoretical_weights.png",
      url: "assets/research_plots/empirical_vs_theoretical_weights.png",
      badge: "Core Discrepancy Insight",
      highlight: "Thinking pauses (18.2% vs 3%) and backspaces (15.4% vs 7%) carry far greater empirical weight than theoretical formulas assumed.",
      description: "Grouped bar chart illustrating the divergence between real-world machine-learning feature importance (XGBoost) and hand-crafted theoretical formula weights across the top 10 telemetry signals."
    },
    {
      id: "plot_modality_validation",
      title: "Multimodal Modality Distributions & 1:1 Ground Truth Validation",
      category: "Formula Validation",
      filename: "modality_distributions_and_formula_validation.png",
      url: "assets/research_plots/modality_distributions_and_formula_validation.png",
      badge: "r = 0.774 Pearson",
      highlight: "Calculated document formula CLI strongly aligns with ground truth dataset score across all 12,000 participant rows.",
      description: "Two-panel visualization: (Left) Boxplot showing normalized distributions of E, F, M, N, and P modalities [0-1]. (Right) Scatter plot comparing the theoretical composite CLI against ground truth with 1:1 identity line."
    },
    {
      id: "plot_unsupervised_gmm",
      title: "Unsupervised GMM Cognitive Load Clusters in 2D Strain Space",
      category: "Unsupervised Clustering",
      filename: "unsupervised_gmm_clusters_2d_strain.png",
      url: "assets/research_plots/unsupervised_gmm_clusters_2d_strain.png",
      badge: "Discovered Clusters",
      highlight: "Separates participants into 4 clean load states (Flow, Moderate, High, Overload) purely from behavioral pauses and ocular fatigue.",
      description: "Scatter plot in 2D Latent Strain Space (Latent Behavioral Strain vs Latent Ocular Fatigue) with viridis color mapping for 4 unsupervised GMM cluster states."
    },
    {
      id: "plot_shap_beeswarm",
      title: "SHAP Beeswarm Plot (Directional Feature Impact)",
      category: "Model Explainability",
      filename: "shap_beeswarm_plot.png",
      url: "assets/research_plots/shap_beeswarm_plot.png",
      badge: "Directional Attribution",
      highlight: "High pause times and elevated backspaces push CLI toward Overload; high EAR values push CLI toward Flow.",
      description: "Beeswarm summary plot showing the distribution of SHAP values for the top 10 predictive features. Red represents high feature values and blue represents low feature values."
    },
    {
      id: "plot_shap_magnitude",
      title: "SHAP Feature Contribution Magnitude (Mean |SHAP|)",
      category: "Model Explainability",
      filename: "shap_feature_importance_bar.png",
      url: "assets/research_plots/shap_feature_importance_bar.png",
      badge: "Top 15 Predictors",
      highlight: "Ocular aspect ratio (min/mean EAR) and keystroke pauses dominate overall cognitive load score variance.",
      description: "Bar plot showing the average absolute contribution (mean |SHAP value|) of each multimodal feature to the predicted cognitive load score."
    },
    {
      id: "plot_shap_distribution",
      title: "SHAP Feature Impact Distribution",
      category: "Model Explainability",
      filename: "shap_feature_impact_distribution.png",
      url: "assets/research_plots/shap_feature_impact_distribution.png",
      badge: "Full Spectrum",
      highlight: "Dense point distribution confirming robust gradient transitions without bifurcated artifacts.",
      description: "Detailed multi-feature SHAP dot plot showing how values across the full test set steer model predictions higher or lower."
    },
    {
      id: "plot_top_15_correlations",
      title: "Top 15 Linear Correlations with CLI Score",
      category: "Linear Correlation",
      filename: "top_15_linear_correlations.png",
      url: "assets/research_plots/top_15_linear_correlations.png",
      badge: "Bivariate Correlation",
      highlight: "min ear (-0.59) and mean ear (-0.56) exhibit the strongest negative correlation; pause seconds (+0.44) exhibits the strongest positive correlation.",
      description: "Horizontal bar chart visualizing Pearson correlation coefficients of the top 15 numeric interaction signals against the reference Cognitive Load Index."
    },
    {
      id: "plot_all_correlations",
      title: "Complete Feature Correlations with CLI Score",
      category: "Linear Correlation",
      filename: "all_feature_correlations.png",
      url: "assets/research_plots/all_feature_correlations.png",
      badge: "Comprehensive Spectrum",
      highlight: "Exhaustive ranking of every collected interaction signal sorted by absolute association strength.",
      description: "Full feature correlation chart highlighting both strong and minor linear associations across the complete telemetry feature set."
    }
  ]
};

let activeTab = 'overview';
let activeMetrics = { ...RESEARCH_METRICS_DATA };

/**
 * Main render function for the research metrics page in the Interaction Logger.
 * @param {string} sourceBack - 'login' or 'dashboard'
 */
export function renderResearchMetricsPage(sourceBack = 'login') {
  const appEl = document.getElementById('app');
  if (!appEl) return;

  // Window scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Load from backend if available
  fetch('/api/analytics/research-metrics')
    .then(res => res.json())
    .then(data => {
      if (data && data.success) {
        activeMetrics = data;
        updateDynamicMetricsContent();
      }
    })
    .catch(() => {
      // Gracefully fall back to the embedded data
    });

  const isLight = document.documentElement.dataset.theme === 'light';

  appEl.innerHTML = `
    <div class="research-page-shell">
      <header class="topbar research-topbar">
        <div class="brand">
          <div class="logo">C</div>
          <div class="brand-text">
            <h1>CogniTrack <span style="color:#38dd9a">AI</span></h1>
            <small>Research Analytics & Model Benchmarks</small>
          </div>
        </div>

        <div class="research-header-meta">
          <span class="research-source-pill">
            <span class="pill-dot">●</span>
            Notebook: <strong>${escapeHtml(activeMetrics.notebook_source)}</strong>
          </span>
          <span class="research-badge-cohort">12,000 Cohort Evaluation</span>
        </div>

        <div class="role-actions research-header-actions">
          <button class="secondary action-btn-compact" id="downloadMetricsJsonBtn" type="button" title="Download raw metric JSON">
            <span aria-hidden="true">⬇</span> Export JSON
          </button>
          <button class="secondary action-btn-compact research-theme-btn" id="researchThemeToggle" type="button" aria-label="Toggle theme">
            ${isLight ? '🌙 Dark Mode' : '☀️ Light Mode'}
          </button>
          <button class="secondary action-btn-compact return-btn" id="backFromResearchBtn" type="button">
            ← Return to ${sourceBack === 'dashboard' ? 'Dashboard' : 'Login'}
          </button>
        </div>
      </header>

      <main class="research-container">
        <!-- Hero Section -->
        <section class="research-hero">
          <div class="eyebrow">INTERACTION LOGGER EMPIRICAL RESEARCH</div>
          <h2>Quantifying Cognitive Load in Human-AI Collaboration</h2>
          <p class="research-hero-desc">
            Exhaustive empirical analysis, SHAP feature attribution, unsupervised 2D strain space modeling,
            and benchmark validation between theoretical formulas and trained machine learning pipelines.
          </p>

          <!-- KPI Metric Strip -->
          <div class="research-kpi-grid">
            <div class="research-kpi-card highlight-green">
              <span class="kpi-label">Winning Model R²</span>
              <strong class="kpi-value">${activeMetrics.winning_architecture.r2_score.toFixed(3)}</strong>
              <small class="kpi-caption">XGBoost Regressor (max_depth=4)</small>
            </div>
            <div class="research-kpi-card highlight-cyan">
              <span class="kpi-label">Strict Tier Accuracy</span>
              <strong class="kpi-value">${activeMetrics.winning_architecture.strict_accuracy.toFixed(1)}%</strong>
              <small class="kpi-caption">Exact 4-class boundary match</small>
            </div>
            <div class="research-kpi-card highlight-purple">
              <span class="kpi-label">Adjacent Safety</span>
              <strong class="kpi-value">${activeMetrics.winning_architecture.adjacent_state_accuracy.toFixed(2)}%</strong>
              <small class="kpi-caption">Zero catastrophic cross-tier leaps</small>
            </div>
            <div class="research-kpi-card highlight-blue">
              <span class="kpi-label">Mean Absolute Error</span>
              <strong class="kpi-value">${activeMetrics.winning_architecture.mae.toFixed(2)} <span class="kpi-unit">pts</span></strong>
              <small class="kpi-caption">Across 0–100 operational scale</small>
            </div>
            <div class="research-kpi-card highlight-emerald">
              <span class="kpi-label">Formula Correlation</span>
              <strong class="kpi-value">r = ${activeMetrics.multimodal_formula.pearson_r.toFixed(3)}</strong>
              <small class="kpi-caption">Pearson r with Ground Truth CLI</small>
            </div>
            <div class="research-kpi-card highlight-amber">
              <span class="kpi-label">Trained Dataset</span>
              <strong class="kpi-value">${Number(activeMetrics.dataset.total_rows).toLocaleString()}</strong>
              <small class="kpi-caption">Participant-task instances</small>
            </div>
          </div>
        </section>

        <!-- Navigation Tabs -->
        <nav class="research-nav-tabs" role="tablist" aria-label="Research Sections">
          <button class="research-tab-btn ${activeTab === 'overview' ? 'active' : ''}" data-tab="overview" role="tab">
            <span class="tab-icon">📊</span> Overview & Model Benchmark
          </button>
          <button class="research-tab-btn ${activeTab === 'weights' ? 'active' : ''}" data-tab="weights" role="tab">
            <span class="tab-icon">⚖️</span> Empirical vs Theoretical Weights
          </button>
          <button class="research-tab-btn ${activeTab === 'multimodal' ? 'active' : ''}" data-tab="multimodal" role="tab">
            <span class="tab-icon">📐</span> Multimodal Formula & Sub-Scores
          </button>
          <button class="research-tab-btn ${activeTab === 'explainability' ? 'active' : ''}" data-tab="explainability" role="tab">
            <span class="tab-icon">🔬</span> SHAP Explainability
          </button>
          <button class="research-tab-btn ${activeTab === 'unsupervised' ? 'active' : ''}" data-tab="unsupervised" role="tab">
            <span class="tab-icon">🌀</span> Unsupervised 2D Strain Clusters
          </button>
          <button class="research-tab-btn ${activeTab === 'tribes' ? 'active' : ''}" data-tab="tribes" role="tab">
            <span class="tab-icon">👥</span> Behavioral Tribes & Correlations
          </button>
        </nav>

        <!-- Tab Content Container -->
        <div id="researchTabContent" class="research-tab-content">
          ${renderTabContent(activeTab)}
        </div>
      </main>

      <!-- Lightbox Modal for Fullscreen Plot Inspection -->
      <div id="plotLightboxModal" class="plot-lightbox-modal hidden" aria-hidden="true">
        <div class="lightbox-overlay" id="lightboxOverlay"></div>
        <div class="lightbox-container card">
          <div class="lightbox-header">
            <div>
              <span class="badge badge-accent" id="lightboxBadge">Visual Plot</span>
              <h3 id="lightboxTitle">Plot Title</h3>
            </div>
            <button class="close-btn" id="closeLightboxBtn" type="button" aria-label="Close dialog">✕</button>
          </div>
          <div class="lightbox-body">
            <img id="lightboxImg" src="" alt="Plot visual" />
          </div>
          <div class="lightbox-footer">
            <p id="lightboxDescription" class="muted-text"></p>
          </div>
        </div>
      </div>
    </div>
  `;

  // Attach interactive events
  attachResearchEvents(sourceBack);
}

function updateDynamicMetricsContent() {
  const tabContent = document.getElementById('researchTabContent');
  if (tabContent) {
    tabContent.innerHTML = renderTabContent(activeTab);
    attachPlotZoomEvents();
  }
}

function renderTabContent(tab) {
  switch (tab) {
    case 'overview':
      return renderOverviewTab();
    case 'weights':
      return renderWeightsTab();
    case 'multimodal':
      return renderMultimodalTab();
    case 'explainability':
      return renderExplainabilityTab();
    case 'unsupervised':
      return renderUnsupervisedTab();
    case 'tribes':
      return renderTribesTab();
    default:
      return renderOverviewTab();
  }
}

// ---------------------------------------------------------
// TAB 1: EXECUTIVE OVERVIEW & MODEL BENCHMARK
// ---------------------------------------------------------
function renderOverviewTab() {
  const win = activeMetrics.winning_architecture;
  const cm = activeMetrics.confusion_matrix;
  const rep = activeMetrics.classification_report;
  const prog = activeMetrics.accuracy_progression;

  return `
    <div class="research-section-grid">
      <!-- Winning Architecture Spotlight -->
      <article class="card panel spotlight-card">
        <div class="section-label">PRODUCTION WINNER</div>
        <div class="spotlight-header">
          <div>
            <h3>${escapeHtml(win.model_name)}</h3>
            <p>${escapeHtml(win.description)}</p>
          </div>
          <span class="badge badge-success nowrap-badge">80.2% Variance Explained (R²)</span>
        </div>

        <div class="metrics-pill-row">
          <div class="metric-pill">
            <span>R² Score</span>
            <strong>${win.r2_score.toFixed(3)}</strong>
          </div>
          <div class="metric-pill">
            <span>Mean Absolute Error</span>
            <strong>${win.mae.toFixed(2)} pts</strong>
          </div>
          <div class="metric-pill">
            <span>Exact Bucket Accuracy</span>
            <strong>${win.strict_accuracy.toFixed(2)}%</strong>
          </div>
          <div class="metric-pill">
            <span>Border-Tolerant (±2.5 pts)</span>
            <strong>${win.border_tolerant_accuracy_2_5.toFixed(2)}%</strong>
          </div>
          <div class="metric-pill">
            <span>Adjacent State Safety</span>
            <strong>${win.adjacent_state_accuracy.toFixed(2)}%</strong>
          </div>
        </div>
      </article>

      <!-- Confusion Matrix Heatmap -->
      <article class="card panel">
        <div class="section-label">ERROR ANALYSIS</div>
        <h3>Confusion Matrix — Boundary Safety Validation</h3>
        <p class="muted-text">
          Evaluation on 2,400 held-out test instances. Classification errors concentrate strictly along adjacent borders with <strong>zero catastrophic jumps</strong>.
        </p>

        <div class="matrix-table-wrap">
          <table class="confusion-table">
            <thead>
              <tr>
                <th class="matrix-corner">Actual State</th>
                <th>Pred Low (&lt;40)</th>
                <th>Pred Mod (40–60)</th>
                <th>Pred High (60–75)</th>
                <th>Pred Overload (≥75)</th>
                <th>Total Support</th>
              </tr>
            </thead>
            <tbody>
              ${cm.matrix.map(row => {
                const total = row.total || (row.pred_low + row.pred_mod + row.pred_high + row.pred_overload);
                return `
                  <tr>
                    <td class="actual-label"><strong>${escapeHtml(row.actual)}</strong></td>
                    <td class="cell ${row.pred_low > 0 ? (row.actual.includes('Low') ? 'hit' : 'miss') : 'zero'}">${row.pred_low}</td>
                    <td class="cell ${row.pred_mod > 0 ? (row.actual.includes('Mod') ? 'hit' : 'miss') : 'zero'}">${row.pred_mod}</td>
                    <td class="cell ${row.pred_high > 0 ? (row.actual.includes('High') ? 'hit' : 'miss') : 'zero'}">${row.pred_high}</td>
                    <td class="cell ${row.pred_overload > 0 ? (row.actual.includes('Overload') ? 'hit' : 'miss') : 'zero'}">${row.pred_overload}</td>
                    <td class="total-cell">${total.toLocaleString()}</td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
        <div class="matrix-callout">
          <span class="callout-icon">🛡️</span>
          <span>${escapeHtml(cm.key_takeaway)}</span>
        </div>
      </article>

      <!-- Model Progression & Benchmark -->
      <article class="card panel">
        <div class="section-label">MODEL PROGRESSION COMPARISON</div>
        <h3>Held-Out Accuracy Progression</h3>
        <div class="monitor-table research-data-table">
          <table>
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th class="text-right">Accuracy</th>
                <th class="text-right">R² Score</th>
                <th class="text-center">Benchmark Status</th>
              </tr>
            </thead>
            <tbody>
              ${prog.map(p => `
                <tr class="${p.status.includes('Winner') ? 'highlight-row' : ''}">
                  <td><strong>${escapeHtml(p.model)}</strong></td>
                  <td class="text-right"><strong>${p.accuracy.toFixed(2)}%</strong></td>
                  <td class="text-right">${p.r2.toFixed(3)}</td>
                  <td class="text-center">
                    <span class="badge ${p.status.includes('Winner') ? 'badge-success' : 'badge-neutral'} nowrap-badge">
                      ${p.status.includes('Winner') ? '★ ' : ''}${escapeHtml(p.status)}
                    </span>
                  </td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </article>

      <!-- Detailed Classification Report -->
      <article class="card panel">
        <div class="section-label">CLASS RECALL & PRECISION</div>
        <h3>Detailed Classification Report</h3>
        <div class="monitor-table research-data-table">
          <table>
            <thead>
              <tr>
                <th>Cognitive Load State</th>
                <th class="text-right">Precision</th>
                <th class="text-right">Recall</th>
                <th class="text-right">F1-Score</th>
                <th class="text-right">Sample Support</th>
              </tr>
            </thead>
            <tbody>
              ${rep.map(r => `
                <tr>
                  <td><strong>${escapeHtml(r.class)}</strong></td>
                  <td class="text-right">${(r.precision * 100).toFixed(1)}%</td>
                  <td class="text-right">${(r.recall * 100).toFixed(1)}%</td>
                  <td class="text-right"><strong style="color:var(--blue)">${(r.f1_score * 100).toFixed(1)}%</strong></td>
                  <td class="text-right muted-text">${r.support.toLocaleString()}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </article>
    </div>
  `;
}

// ---------------------------------------------------------
// TAB 2: EMPIRICAL VS THEORETICAL WEIGHTS
// ---------------------------------------------------------
function renderWeightsTab() {
  const plot = activeMetrics.plots.find(p => p.id === 'plot_empirical_vs_theoretical');
  const weights = activeMetrics.weights_comparison;

  return `
    <div class="research-section-grid">
      <!-- Plot Card -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(plot.category)}</div>
            <h3>${escapeHtml(plot.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(plot.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(plot.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${plot.id}">
          <img src="${plot.url}" alt="${escapeHtml(plot.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>

        <p class="muted-text">${escapeHtml(plot.description)}</p>
      </article>

      <!-- Discrepancy Breakdown Table -->
      <article class="card panel">
        <div class="section-label">EMPIRICAL VS THEORETICAL ANALYSIS</div>
        <h3>Top 10 Feature Weights Comparison</h3>
        <p class="muted-text">${escapeHtml(weights.description)}</p>

        <div class="monitor-table research-data-table">
          <table>
            <thead>
              <tr>
                <th>Interaction Signal</th>
                <th>Modality</th>
                <th class="text-right">Empirical ML (%)</th>
                <th class="text-right">Theoretical Doc (%)</th>
                <th class="text-center">Divergence Delta</th>
              </tr>
            </thead>
            <tbody>
              ${weights.features.map(f => {
                const isPos = f.difference.startsWith('+');
                return `
                  <tr>
                    <td>
                      <div class="signal-name-cell">
                        <strong>${escapeHtml(f.name || f.feature)}</strong>
                        <small class="code-subtext">${escapeHtml(f.code || f.feature)}</small>
                      </div>
                    </td>
                    <td><span class="modality-tag">${escapeHtml(f.modality)}</span></td>
                    <td class="text-right"><strong style="color:var(--blue)">${f.empirical_pct.toFixed(2)}%</strong></td>
                    <td class="text-right muted-text">${f.theoretical_pct.toFixed(2)}%</td>
                    <td class="text-center">
                      <span class="delta-badge ${isPos ? 'delta-pos' : 'delta-neg'}">${escapeHtml(f.difference)}</span>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>

        <div class="research-insight-box">
          <h4>💡 Key Research Discovery</h4>
          <p>
            Real user interactions show that <strong>thinking pauses (18.23%)</strong> and <strong>backspace corrections (15.42%)</strong>
            are substantially stronger predictors of mental load than pure theoretical models assumed (3% and 7% respectively).
            In human-AI collaboration, cognitive strain surfaces primarily through verbal hesitation and drafting friction before visual fatigue sets in.
          </p>
        </div>
      </article>
    </div>
  `;
}

// ---------------------------------------------------------
// TAB 3: MULTIMODAL FORMULA & SUB-SCORES
// ---------------------------------------------------------
function renderMultimodalTab() {
  const plot = activeMetrics.plots.find(p => p.id === 'plot_modality_validation');
  const formula = activeMetrics.multimodal_formula;

  return `
    <div class="research-section-grid">
      <!-- Formula Definition Card -->
      <article class="card panel formula-banner-card">
        <div class="section-label">MATHEMATICAL FORMULATION</div>
        <h3>Proposed Multimodal Cognitive Load Formula</h3>
        
        <div class="formula-box-container">
          <div class="formula-display-code">
            ${escapeHtml(formula.formula)}
          </div>
        </div>

        <div class="formula-legend-grid">
          <div class="legend-item"><span class="legend-badge">E (30%)</span><span>Eye Tracking / Ocular Strain</span></div>
          <div class="legend-item"><span class="legend-badge">F (10%)</span><span>Facial Expression / Frustration</span></div>
          <div class="legend-item"><span class="legend-badge">M (15%)</span><span>Mouse Dynamics & Pauses</span></div>
          <div class="legend-item"><span class="legend-badge">N (25%)</span><span>NASA-TLX Subjective Load</span></div>
          <div class="legend-item"><span class="legend-badge">P (20%)</span><span>Task Performance & Accuracy</span></div>
        </div>

        <div class="metrics-pill-row" style="margin-top:1.5rem;">
          <div class="metric-pill">
            <span>Pearson Correlation</span>
            <strong>r = ${formula.pearson_r.toFixed(3)}</strong>
          </div>
          <div class="metric-pill">
            <span>Spearman Rank</span>
            <strong>ρ = ${formula.spearman_rho.toFixed(3)}</strong>
          </div>
          <div class="metric-pill">
            <span>Mean Absolute Difference</span>
            <strong>${formula.mae_points.toFixed(2)} pts</strong>
          </div>
        </div>
      </article>

      <!-- Plot Visual Card -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(plot.category)}</div>
            <h3>${escapeHtml(plot.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(plot.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(plot.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${plot.id}">
          <img src="${plot.url}" alt="${escapeHtml(plot.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(plot.description)}</p>
      </article>

      <!-- Modalities Table -->
      <article class="card panel full-width-card">
        <div class="section-label">COMPONENT SUB-SCORES (0 TO 1 SCALE)</div>
        <h3>Normalized Modality Metrics Across 12,000 Cohort</h3>
        <div class="monitor-table research-data-table">
          <table>
            <thead>
              <tr>
                <th>Modality Sub-Score</th>
                <th class="text-center">Formula Weight</th>
                <th class="text-right">Mean</th>
                <th class="text-right">Std Dev</th>
                <th class="text-center">Operational Range</th>
                <th>Underlying Telemetry Signals</th>
              </tr>
            </thead>
            <tbody>
              ${formula.sub_scores.map(s => `
                <tr>
                  <td><strong>${escapeHtml(s.modality)}</strong></td>
                  <td class="text-center"><span class="badge badge-neutral nowrap-badge">${escapeHtml(s.weight)}</span></td>
                  <td class="text-right"><strong>${s.mean.toFixed(3)}</strong></td>
                  <td class="text-right muted-text">±${s.std.toFixed(3)}</td>
                  <td class="text-center"><span class="range-badge">${s.min.toFixed(3)} – ${s.max.toFixed(3)}</span></td>
                  <td><span class="signals-desc">${escapeHtml(s.components)}</span></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </article>
    </div>
  `;
}

// ---------------------------------------------------------
// TAB 4: SHAP EXPLAINABILITY
// ---------------------------------------------------------
function renderExplainabilityTab() {
  const beeswarm = activeMetrics.plots.find(p => p.id === 'plot_shap_beeswarm');
  const magnitude = activeMetrics.plots.find(p => p.id === 'plot_shap_magnitude');
  const distribution = activeMetrics.plots.find(p => p.id === 'plot_shap_distribution');

  return `
    <div class="research-section-grid">
      <!-- Beeswarm Plot Card -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(beeswarm.category)}</div>
            <h3>${escapeHtml(beeswarm.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(beeswarm.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(beeswarm.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${beeswarm.id}">
          <img src="${beeswarm.url}" alt="${escapeHtml(beeswarm.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(beeswarm.description)}</p>
      </article>

      <!-- Magnitude Plot Card -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(magnitude.category)}</div>
            <h3>${escapeHtml(magnitude.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(magnitude.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(magnitude.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${magnitude.id}">
          <img src="${magnitude.url}" alt="${escapeHtml(magnitude.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(magnitude.description)}</p>
      </article>

      <!-- Feature Impact Distribution Plot Card -->
      <article class="card panel plot-feature-card full-width-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(distribution.category)}</div>
            <h3>${escapeHtml(distribution.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(distribution.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(distribution.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${distribution.id}">
          <img src="${distribution.url}" alt="${escapeHtml(distribution.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(distribution.description)}</p>
      </article>
    </div>
  `;
}

// ---------------------------------------------------------
// TAB 5: UNSUPERVISED GMM STRAIN CLUSTERS
// ---------------------------------------------------------
function renderUnsupervisedTab() {
  const plot = activeMetrics.plots.find(p => p.id === 'plot_unsupervised_gmm');
  const gmm = activeMetrics.unsupervised_gmm_clusters;

  return `
    <div class="research-section-grid">
      <!-- Plot Card -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(plot.category)}</div>
            <h3>${escapeHtml(plot.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(plot.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(plot.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${plot.id}">
          <img src="${plot.url}" alt="${escapeHtml(plot.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(plot.description)}</p>
      </article>

      <!-- GMM Cluster Profiles -->
      <article class="card panel">
        <div class="section-label">UNSUPERVISED CLUSTER PROFILES</div>
        <h3>Discovered 4 GMM Cognitive Load Clusters</h3>
        <p class="muted-text">${escapeHtml(gmm.description)}</p>

        <div class="metrics-pill-row">
          <div class="metric-pill">
            <span>Silhouette Score</span>
            <strong>${gmm.silhouette_score.toFixed(3)}</strong>
          </div>
          <div class="metric-pill">
            <span>Davies-Bouldin Index</span>
            <strong>${gmm.davies_bouldin_index.toFixed(3)}</strong>
          </div>
          <div class="metric-pill">
            <span>Spearman with CLI</span>
            <strong>${gmm.spearman_correlation_ground_truth.toFixed(3)}</strong>
          </div>
        </div>

        <div class="monitor-table research-data-table" style="margin-top:16px;">
          <table>
            <thead>
              <tr>
                <th>Discovered Cluster</th>
                <th class="text-right">Users</th>
                <th class="text-right">Pause (s)</th>
                <th class="text-right">Backspace</th>
                <th class="text-right">PERCLOS</th>
                <th class="text-right">Unsupervised CLI</th>
                <th class="text-right">Ground Truth</th>
              </tr>
            </thead>
            <tbody>
              ${gmm.clusters.map(c => `
                <tr>
                  <td>
                    <div class="cluster-cell">
                      <strong>${escapeHtml(c.cluster)}</strong>
                      <small class="cluster-desc">${escapeHtml(c.strain_profile)}</small>
                    </div>
                  </td>
                  <td class="text-right muted-text">${c.count.toLocaleString()}</td>
                  <td class="text-right">${c.pause_sec.toFixed(2)}s</td>
                  <td class="text-right">${c.backspace_rate.toFixed(2)}</td>
                  <td class="text-right">${c.mean_perclos.toFixed(2)}</td>
                  <td class="text-right"><strong style="color:var(--blue)">${c.unsup_cli.toFixed(2)}</strong></td>
                  <td class="text-right"><strong style="color:var(--text)">${c.actual_cli_score.toFixed(2)}</strong></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </article>

      <!-- Derived PCA Weights -->
      <article class="card panel full-width-card">
        <div class="section-label">DATA-DRIVEN FORMULA DERIVATION</div>
        <h3>Unsupervised PCA Formula Weights</h3>
        <p class="muted-text">
          Weights derived purely from explained variance ratios without target supervision.
        </p>
        <div class="monitor-table research-data-table">
          <table>
            <thead>
              <tr>
                <th>Feature Name</th>
                <th>Modality</th>
                <th class="text-right">Derived Weight (%)</th>
              </tr>
            </thead>
            <tbody>
              ${gmm.derived_pca_weights.map(w => `
                <tr>
                  <td>
                    <div class="signal-name-cell">
                      <strong>${escapeHtml(w.name || w.feature)}</strong>
                      <small class="code-subtext">${escapeHtml(w.code || w.feature)}</small>
                    </div>
                  </td>
                  <td><span class="modality-tag">${escapeHtml(w.modality)}</span></td>
                  <td class="text-right"><strong style="color:var(--blue)">${w.weight_pct.toFixed(2)}%</strong></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        </div>
      </article>
    </div>
  `;
}

// ---------------------------------------------------------
// TAB 6: BEHAVIORAL TRIBES & LINEAR CORRELATIONS
// ---------------------------------------------------------
function renderTribesTab() {
  const tribes = activeMetrics.behavioral_tribes;
  const corrs = activeMetrics.linear_correlations;
  const pTop15 = activeMetrics.plots.find(p => p.id === 'plot_top_15_correlations');
  const pAll = activeMetrics.plots.find(p => p.id === 'plot_all_correlations');

  return `
    <div class="research-section-grid">
      <!-- Behavioral Tribes Spotlight -->
      <article class="card panel full-width-card">
        <div class="section-label">UNSUPERVISED K-MEANS BEHAVIORAL TRIBES</div>
        <h3>4 Natural Participant Clusters in Human-AI Interaction</h3>
        <p class="muted-text">
          Discovered by clustering 12,000 participant sessions along pause duration, backspace rate, and keystroke burst volume.
        </p>

        <div class="tribes-card-grid">
          ${tribes.map(t => `
            <div class="tribe-card card">
              <div class="tribe-badge">Tribe ${t.tribe_id}</div>
              <h4>${escapeHtml(t.label)}</h4>
              <p class="tribe-desc">${escapeHtml(t.description)}</p>
              <div class="tribe-stats">
                <div><span>Cohort Share</span><strong>${t.pct.toFixed(1)}% (${t.user_count.toLocaleString()})</strong></div>
                <div><span>Avg Pause</span><strong>${t.avg_pause_sec.toFixed(2)}s</strong></div>
                <div><span>Backspace Rate</span><strong>${t.avg_backspace_rate.toFixed(2)}</strong></div>
                <div><span>Avg Keystrokes</span><strong>${t.avg_total_keys.toFixed(1)}</strong></div>
                <div><span>Mean CLI</span><strong style="color:var(--blue)">${t.avg_cli.toFixed(2)}</strong></div>
              </div>
            </div>
          `).join('')}
        </div>
      </article>

      <!-- Top 15 Correlations Plot -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(pTop15.category)}</div>
            <h3>${escapeHtml(pTop15.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(pTop15.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(pTop15.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${pTop15.id}">
          <img src="${pTop15.url}" alt="${escapeHtml(pTop15.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(pTop15.description)}</p>
      </article>

      <!-- All Feature Correlations Plot -->
      <article class="card panel plot-feature-card">
        <div class="plot-header">
          <div>
            <div class="section-label">${escapeHtml(pAll.category)}</div>
            <h3>${escapeHtml(pAll.title)}</h3>
          </div>
          <span class="badge badge-accent nowrap-badge">${escapeHtml(pAll.badge)}</span>
        </div>
        <p class="plot-highlight-text">${escapeHtml(pAll.highlight)}</p>

        <div class="plot-image-container" data-zoom-plot="${pAll.id}">
          <img src="${pAll.url}" alt="${escapeHtml(pAll.title)}" class="research-plot-img" />
          <div class="plot-zoom-overlay">
            <span>🔍 Click to Expand High-Resolution Chart</span>
          </div>
        </div>
        <p class="muted-text">${escapeHtml(pAll.description)}</p>
      </article>

      <!-- Correlation Rankings Table -->
      <article class="card panel full-width-card">
        <div class="section-label">TELEMETRY CORRELATION MATRIX</div>
        <h3>Ranked Linear Correlation Coefficients</h3>
        <div class="monitor-table research-data-table">
          <table>
            <thead>
              <tr>
                <th>Telemetry Signal</th>
                <th>Category</th>
                <th class="text-right">Correlation (r)</th>
                <th class="text-center">Direction</th>
                <th>Physiological / Interaction Rationale</th>
              </tr>
            </thead>
            <tbody>
              ${corrs.map(c => {
                const isPos = c.corr > 0;
                return `
                  <tr>
                    <td>
                      <div class="signal-name-cell">
                        <strong>${escapeHtml(c.name || c.feature)}</strong>
                        <small class="code-subtext">${escapeHtml(c.code || c.feature)}</small>
                      </div>
                    </td>
                    <td><span class="modality-tag">${escapeHtml(c.type)}</span></td>
                    <td class="text-right">
                      <strong style="color:${isPos ? 'var(--danger)' : 'var(--green)'}">
                        ${(c.corr > 0 ? '+' : '') + c.corr.toFixed(3)}
                      </strong>
                    </td>
                    <td class="text-center">
                      <span class="badge ${isPos ? 'badge-danger' : 'badge-success'} nowrap-badge">${escapeHtml(c.direction)}</span>
                    </td>
                    <td><span class="rationale-text">${escapeHtml(c.note)}</span></td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </article>
    </div>
  `;
}

// ---------------------------------------------------------
// EVENT HANDLERS & LIGHTBOX
// ---------------------------------------------------------
function attachResearchEvents(sourceBack) {
  // Return button
  const backBtn = document.getElementById('backFromResearchBtn');
  if (backBtn) {
    backBtn.onclick = () => {
      if (sourceBack === 'dashboard') {
        if (typeof window.renderIndependentDashboard === 'function') {
          window.renderIndependentDashboard(sessionStorage.getItem('cognitrack_dashboard_mode') || 'group');
        } else if (typeof window.renderRoleDashboard === 'function') {
          window.renderRoleDashboard('admin');
        } else if (typeof window.renderLogin === 'function') {
          window.renderLogin();
        }
      } else {
        if (typeof window.renderLogin === 'function') {
          window.renderLogin();
        } else {
          window.location.reload();
        }
      }
    };
  }

  // Theme toggle button
  const themeToggle = document.getElementById('researchThemeToggle');
  if (themeToggle) {
    themeToggle.onclick = () => {
      const current = document.documentElement.dataset.theme;
      const next = current === 'light' ? 'dark' : 'light';
      document.documentElement.dataset.theme = next;
      localStorage.setItem('cognitrack_theme_v3', next);
      themeToggle.textContent = next === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode';
    };
  }

  // Download raw metrics JSON
  const downloadBtn = document.getElementById('downloadMetricsJsonBtn');
  if (downloadBtn) {
    downloadBtn.onclick = () => {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(activeMetrics, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", "cognitrack_research_metrics.json");
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    };
  }

  // Tab switching
  const tabBtns = document.querySelectorAll('.research-tab-btn');
  tabBtns.forEach(btn => {
    btn.onclick = () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTab = btn.dataset.tab;
      const container = document.getElementById('researchTabContent');
      if (container) {
        container.innerHTML = renderTabContent(activeTab);
        attachPlotZoomEvents();
      }
    };
  });

  // Lightbox close button
  const closeBtn = document.getElementById('closeLightboxBtn');
  const overlay = document.getElementById('lightboxOverlay');
  const modal = document.getElementById('plotLightboxModal');

  const closeModal = () => {
    if (modal) {
      modal.classList.add('hidden');
      modal.setAttribute('aria-hidden', 'true');
    }
  };

  if (closeBtn) closeBtn.onclick = closeModal;
  if (overlay) overlay.onclick = closeModal;

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal && !modal.classList.contains('hidden')) {
      closeModal();
    }
  });

  // Zoom events for plots
  attachPlotZoomEvents();
}

function attachPlotZoomEvents() {
  const zoomTriggers = document.querySelectorAll('[data-zoom-plot]');
  const modal = document.getElementById('plotLightboxModal');
  const titleEl = document.getElementById('lightboxTitle');
  const badgeEl = document.getElementById('lightboxBadge');
  const imgEl = document.getElementById('lightboxImg');
  const descEl = document.getElementById('lightboxDescription');

  zoomTriggers.forEach(trigger => {
    trigger.onclick = () => {
      const plotId = trigger.dataset.zoomPlot;
      const plot = activeMetrics.plots.find(p => p.id === plotId);
      if (!plot || !modal) return;

      if (titleEl) titleEl.textContent = plot.title;
      if (badgeEl) badgeEl.textContent = plot.badge || plot.category;
      if (imgEl) {
        imgEl.src = plot.url;
        imgEl.alt = plot.title;
      }
      if (descEl) descEl.textContent = plot.description;

      modal.classList.remove('hidden');
      modal.setAttribute('aria-hidden', 'false');
    };
  });
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// Expose globally for vanilla browser scripts
if (typeof window !== 'undefined') {
  window.renderResearchMetricsPage = renderResearchMetricsPage;
}
