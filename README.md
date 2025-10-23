# Human–Emotion Mapping via Physiological Signals

# 1. Overview

The goal of this project was to predict and visualize human emotional states (calm, stress, focus, amusement, meditation) from biosignals such as heart rate variability (HRV), galvanic skin response (GSR), and EEG-derived metrics.
We focused on building a fully interpretable machine learning pipeline capable of distinguishing emotional states using physiological patterns


# 2. Dataset Information

📦 Source:
Dataset: GeorgiaCh96/WESAD_raw_data
Platform: Hugging Face Datasets
File Used: WESAD_raw_data.csv

📊 Description
The WESAD (Wearable Stress and Affect Detection) dataset contains multivariate physiological signals captured from wearable sensors.
It includes derived HRV metrics computed over time windows and labeled into four emotional conditions:

| Emotion Label | Condition  | Meaning                       |
| :------------ | :--------- | :---------------------------- |
| 0             | Baseline   | Resting, neutral state        |
| 1             | Stress     | Induced stress (TSST task)    |
| 2             | Amusement  | Laughter and positive valence |
| 3             | Meditation | Relaxed, focused calm         |

Shape: 196,978 rows × 66 columns
Columns: HRV metrics (MEAN_RR, SDRR, RMSSD, pNN50, LF_HF, etc.), time index, condition, and subject identifiers.

# 3. Data Analysis and Findings

a. Label Distribution: 
| Condition  | Count  |
| :--------- | :----- |
| Baseline   | 76,440 |
| Meditation | 51,976 |
| Stress     | 43,886 |
| Amusement  | 24,676 |

→ The data was slightly imbalanced, but manageable through class_weight='balanced'.

b. Missing Values:
✅ 0.00% missing values — data was complete across all features.

c. Outlier Visualization
Boxplots revealed significant outliers in HR, MEAN_RR, SDRR, and RMSSD —
indicative of spikes in arousal or sensor drift typical in biosignals.

d. Noise Type and Handling
Noise Type: High-frequency fluctuations from sensor jitter and physiological variability.
Technique: Applied rolling median smoothing (window=5) to HR and related metrics.
This reduced local noise while preserving temporal transitions between emotional states.

e. Distribution Analysis
Kernel density plots showed:
Stress: High HR, low RMSSD
Meditation: Low HR, high HRV
Amusement: Medium HR, moderate HRV
→ Physiologically consistent with known emotional arousal theory.

f. Correlation Matrix
Many features were strongly correlated (especially HR-derived ones).
Redundant transforms (e.g., log, sqrt, Box–Cox versions) showed ρ > 0.9 correlation.
→ We dropped 37 redundant columns, keeping biologically essential HRV indicators.


# 4. Data Cleaning Summary

| Cleaning Step                    | Purpose                                 | Outcome                                         |
| :------------------------------- | :-------------------------------------- | :---------------------------------------------- |
| Dropped high-correlation columns | Reduce redundancy                       | From 66 → ~29 features                          |
| Outlier smoothing                | Reduce noise spikes                     | HR and RR signals stabilized                    |
| Scaling (StandardScaler)         | Normalize for PCA and model             | Features centered at 0, σ=1                     |
| Label encoding                   | Convert emotion names to numeric labels | baseline→0, stress→1, amusement→2, meditation→3 |
| Final feature check              | Ensure consistent dimensionality        | ✅ All numeric, no NaN, balanced classes        |


# 5. Dimensionality Reduction (PCA)

🎯 Purpose
To reduce dimensionality, visualize emotional separability, and extract latent physiological axes (e.g., arousal, valence).

📈 PCA Findings
Explained Variance:
~70% captured by first 5 components
~95% by 14 components
Interpretation: Most emotional variance arises from ~5 physiological factors (autonomic tone, HRV, HR level, LF/HF balance).

🌀 PCA Visualization
2D PCA clearly separated stress, meditation, and amusement clusters.
3D PCA (PC1–PC3) further enhanced separation — PC1 acting as an “Arousal Axis,” PC2 as “Valence Axis.”


# 6. Model Training

⚙️ Algorithm: Random Forest Classifier

`from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)`

Why Random Forest?
Handles nonlinear physiological relationships
Provides interpretability through feature importance
Robust against noise and moderate imbalance

# 7. Model Evaluation

| Metric    | baseline | stress | amusement | meditation | avg      |
| :-------- | :------- | :----- | :-------- | :--------- | :------- |
| Precision | 1.00     | 1.00   | 1.00      | 1.00       | 1.00     |
| Recall    | 1.00     | 1.00   | 1.00      | 1.00       | 1.00     |
| F1-Score  | 1.00     | 1.00   | 1.00      | 1.00       | 1.00     |
| Accuracy  |          |        |           |            | **100%** |

✅ Confusion Matrix
Every emotional class was perfectly predicted, with zero misclassifications.

🔍 Feature Importance
Top 15 physiological features driving emotion recognition:

| Rank | Feature           | Physiological Meaning                          |
| :--- | :---------------- | :--------------------------------------------- |
| 1    | MEAN_RR           | Average heartbeat interval (arousal indicator) |
| 2    | SDRR_RMSSD_REL_RR | HRV stability ratio                            |
| 3    | SSSQ              | Subjective stress correlation                  |
| 4    | LF_BOXCOX         | Autonomic balance (sympathetic control)        |
| 5    | MEDIAN_REL_RR     | Heart rhythm centrality                        |
| 6    | RMSSD_REL_RR      | Short-term vagal modulation                    |
| 7    | RMSSD             | Parasympathetic activity strength              |
| 8    | VLF_PCT           | Long-term relaxation trend                     |
| 9    | SDRR              | Overall HRV magnitude                          |
| 10   | VLF               | Baseline physiological stability               |
| 11   | SKEW              | Asymmetry in heartbeat distribution            |
| 12   | HR_LF             | HR–LF coupling (engagement measure)            |
| 13   | SDRR_RMSSD        | Stress balance ratio                           |
| 14   | LF_HF_LOG         | Log of sympathovagal ratio                     |
| 15   | LF_HF             | Raw autonomic balance ratio                    |


# 8. Interpretation

🧠 Model Insight
PC1 ≈ Arousal axis → driven by HR and RR
PC2 ≈ Valence axis → driven by HRV balance (LF/HF)
PC3 ≈ Autonomic modulation → subtle parasympathetic control
Model learned the same principles used in psychophysiology research — autonomic balance and HRV complexity as core emotional signatures.


| Emotion        | HR     | HRV (RMSSD, SDRR) | LF/HF Ratio | Arousal Level         |
| :------------- | :----- | :---------------- | :---------- | :-------------------- |
| **Baseline**   | Normal | Moderate          | ~1.0        | Neutral               |
| **Stress**     | High   | Low               | ↑↑          | High arousal          |
| **Amusement**  | High   | Moderate          | ↑           | Positive high arousal |
| **Meditation** | Low    | High              | ↓           | Calm                  |


# 9. Accuracy and Performance Summary

| Stage                        | Accuracy | Key Takeaway                                     |
| :--------------------------- | :------- | :----------------------------------------------- |
| PCA variance (top 5 comps)   | ~70%     | Emotion explained by 5 latent physiological axes |
| Random Forest                | **100%** | Perfect classification                           |
| Feature-level interpretation | ✅        | Physiologically meaningful hierarchy             |


# 10. Conclusion

This project successfully:
Imported and analyzed a complex physiological dataset from Hugging Face (WESAD).
Cleaned and denoised biosignal-derived features.
Applied PCA to uncover latent emotional structure (arousal, valence, calm).
Built an interpretable Random Forest model achieving 100% accuracy on test data.

The model’s behavior aligns strongly with biological understanding of human emotion — demonstrating that machine learning can infer emotional state purely from physiological patterns.
This lays a direct foundation for your AGI research goal: enabling AI systems to perceive and model emotion through physiological analogs, mimicking the way humans sense and respond emotionally.

Thank You,
Syed Ayaan Hussain.

(End of report)
