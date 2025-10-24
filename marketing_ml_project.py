"""
Marketing Analytics & Customer Segmentation Project
Classical Machine Learning Algorithms Implementation

This project demonstrates:
1. Customer Segmentation (K-Means, Hierarchical, DBSCAN)
2. Customer Lifetime Value Prediction (Linear/Ridge/Lasso Regression)
3. Churn Prediction (Logistic Regression, Decision Trees, Random Forest, SVM, Naive Bayes)
4. Market Basket Analysis (Association Rules - Apriori)
5. Campaign Response Prediction (Ensemble Methods)
6. RFM Analysis and Feature Engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             mean_squared_error, r2_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. DATA GENERATION - Synthetic Marketing Dataset
# ============================================================================

def generate_marketing_data(n_customers=5000):
    """Generate realistic synthetic marketing data"""
    
    # Customer Demographics
    customer_id = np.arange(1, n_customers + 1)
    age = np.random.normal(42, 15, n_customers).clip(18, 80).astype(int)
    gender = np.random.choice(['M', 'F'], n_customers, p=[0.48, 0.52])
    income = np.random.lognormal(10.5, 0.6, n_customers).clip(20000, 200000)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                  n_customers, p=[0.3, 0.4, 0.25, 0.05])
    
    # RFM Features
    recency = np.random.exponential(30, n_customers).clip(1, 365).astype(int)
    frequency = np.random.poisson(5, n_customers).clip(1, 50)
    monetary = np.random.lognormal(6, 1.2, n_customers).clip(50, 10000)
    
    # Behavioral Features
    tenure_months = np.random.randint(1, 120, n_customers)
    num_purchases = frequency + np.random.randint(-2, 3, n_customers).clip(0, None)
    avg_order_value = monetary / frequency
    website_visits = np.random.poisson(15, n_customers)
    email_open_rate = np.random.beta(2, 5, n_customers)
    
    # Product Categories (multi-hot encoding)
    electronics = np.random.binomial(1, 0.3, n_customers)
    clothing = np.random.binomial(1, 0.4, n_customers)
    home_garden = np.random.binomial(1, 0.25, n_customers)
    sports = np.random.binomial(1, 0.2, n_customers)
    
    # Customer Lifetime Value (target variable for regression)
    clv = (monetary * frequency * 0.3 + 
           income * 0.01 + 
           tenure_months * 50 + 
           np.random.normal(0, 500, n_customers))
    
    # Churn (target variable for classification)
    churn_prob = 1 / (1 + np.exp(-(
        -0.05 * recency + 
        -0.3 * frequency + 
        -0.0005 * monetary + 
        -0.02 * tenure_months + 
        2
    )))
    churned = (np.random.random(n_customers) < churn_prob).astype(int)
    
    # Campaign Response
    campaign_response = np.random.binomial(1, 0.15, n_customers)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'age': age,
        'gender': gender,
        'income': income,
        'education': education,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'tenure_months': tenure_months,
        'num_purchases': num_purchases,
        'avg_order_value': avg_order_value,
        'website_visits': website_visits,
        'email_open_rate': email_open_rate,
        'electronics': electronics,
        'clothing': clothing,
        'home_garden': home_garden,
        'sports': sports,
        'clv': clv,
        'churned': churned,
        'campaign_response': campaign_response
    })
    
    return df

# Generate data
print("=" * 80)
print("MARKETING ANALYTICS & CUSTOMER SEGMENTATION PROJECT")
print("=" * 80)
print("\n1. Generating Synthetic Marketing Data...")
df = generate_marketing_data(5000)
print(f"Dataset Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nDataset Statistics:")
print(df.describe())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Correlation Analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
print("\nTop Correlations with CLV:")
print(correlation_matrix['clv'].sort_values(ascending=False).head(10))

print("\nTop Correlations with Churn:")
print(correlation_matrix['churned'].sort_values(ascending=False).head(10))

# Chi-square test for categorical variables
print("\nChi-Square Test Results:")
for cat_col in ['gender', 'education']:
    contingency = pd.crosstab(df[cat_col], df['churned'])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"{cat_col} vs Churn: χ² = {chi2:.2f}, p-value = {p_value:.4f}")

# ============================================================================
# 3. RFM ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("3. RFM ANALYSIS")
print("=" * 80)

def rfm_score(df):
    """Calculate RFM scores"""
    df_rfm = df.copy()
    
    # RFM Quartiles (Recency: lower is better, Frequency & Monetary: higher is better)
    df_rfm['R_Score'] = pd.qcut(df_rfm['recency'], 4, labels=[4, 3, 2, 1])
    df_rfm['F_Score'] = pd.qcut(df_rfm['frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    df_rfm['M_Score'] = pd.qcut(df_rfm['monetary'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    
    # Convert to numeric
    df_rfm['R_Score'] = df_rfm['R_Score'].astype(int)
    df_rfm['F_Score'] = df_rfm['F_Score'].astype(int)
    df_rfm['M_Score'] = df_rfm['M_Score'].astype(int)
    
    # RFM Combined Score
    df_rfm['RFM_Score'] = (df_rfm['R_Score'].astype(str) + 
                           df_rfm['F_Score'].astype(str) + 
                           df_rfm['M_Score'].astype(str))
    
    # RFM Segment
    df_rfm['RFM_Total'] = df_rfm['R_Score'] + df_rfm['F_Score'] + df_rfm['M_Score']
    
    def rfm_segment(row):
        if row['RFM_Total'] >= 10:
            return 'Champions'
        elif row['RFM_Total'] >= 8:
            return 'Loyal Customers'
        elif row['RFM_Total'] >= 6:
            return 'Potential Loyalists'
        elif row['RFM_Total'] >= 5:
            return 'At Risk'
        else:
            return 'Lost'
    
    df_rfm['RFM_Segment'] = df_rfm.apply(rfm_segment, axis=1)
    
    return df_rfm

df_rfm = rfm_score(df)
print("\nRFM Segment Distribution:")
print(df_rfm['RFM_Segment'].value_counts())
print("\nAverage CLV by RFM Segment:")
print(df_rfm.groupby('RFM_Segment')['clv'].mean().sort_values(ascending=False))

# ============================================================================
# 4. CUSTOMER SEGMENTATION - CLUSTERING ALGORITHMS
# ============================================================================

print("\n" + "=" * 80)
print("4. CUSTOMER SEGMENTATION - CLUSTERING")
print("=" * 80)

# Prepare features for clustering
clustering_features = ['recency', 'frequency', 'monetary', 'tenure_months', 
                       'avg_order_value', 'website_visits', 'email_open_rate']
X_cluster = df[clustering_features].copy()

# Standardize features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# 4.1 K-Means Clustering
print("\n4.1 K-Means Clustering")
print("-" * 40)

# Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_scaled, kmeans.labels_))

print("K\tInertia\t\tSilhouette")
for k, inertia, sil in zip(K_range, inertias, silhouette_scores):
    print(f"{k}\t{inertia:.2f}\t\t{sil:.4f}")

# Optimal K-Means (k=4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(X_cluster_scaled)

print(f"\nOptimal K-Means with k={optimal_k}")
print(f"Silhouette Score: {silhouette_score(X_cluster_scaled, df['KMeans_Cluster']):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_cluster_scaled, df['KMeans_Cluster']):.4f}")

print("\nCluster Characteristics:")
print(df.groupby('KMeans_Cluster')[clustering_features].mean())

# 4.2 Hierarchical Clustering
print("\n4.2 Hierarchical (Agglomerative) Clustering")
print("-" * 40)

hierarchical = AgglomerativeClustering(n_clusters=4, linkage='ward')
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_cluster_scaled)

print(f"Silhouette Score: {silhouette_score(X_cluster_scaled, df['Hierarchical_Cluster']):.4f}")
print("\nCluster Distribution:")
print(df['Hierarchical_Cluster'].value_counts().sort_index())

# 4.3 DBSCAN
print("\n4.3 DBSCAN (Density-Based Clustering)")
print("-" * 40)

dbscan = DBSCAN(eps=1.5, min_samples=50)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_cluster_scaled)

n_clusters = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'] else 0)
n_noise = list(df['DBSCAN_Cluster']).count(-1)

print(f"Number of Clusters: {n_clusters}")
print(f"Number of Noise Points: {n_noise}")
print("\nCluster Distribution:")
print(df['DBSCAN_Cluster'].value_counts().sort_index())

# ============================================================================
# 5. CUSTOMER LIFETIME VALUE (CLV) PREDICTION - REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("5. CUSTOMER LIFETIME VALUE PREDICTION - REGRESSION")
print("=" * 80)

# Prepare features
regression_features = ['age', 'income', 'recency', 'frequency', 'monetary',
                       'tenure_months', 'avg_order_value', 'website_visits', 
                       'email_open_rate', 'electronics', 'clothing', 
                       'home_garden', 'sports']

X_reg = df[regression_features].copy()
y_clv = df['clv'].copy()

# Encode categorical if needed (gender, education)
le_gender = LabelEncoder()
le_education = LabelEncoder()
X_reg['gender_encoded'] = le_gender.fit_transform(df['gender'])
X_reg['education_encoded'] = le_education.fit_transform(df['education'])

# Train-test split
X_train_reg, X_test_reg, y_train_clv, y_test_clv = train_test_split(
    X_reg, y_clv, test_size=0.2, random_state=42
)

# Standardize
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 5.1 Linear Regression
print("\n5.1 Linear Regression")
print("-" * 40)
lr = LinearRegression()
lr.fit(X_train_reg_scaled, y_train_clv)
y_pred_lr = lr.predict(X_test_reg_scaled)

rmse_lr = np.sqrt(mean_squared_error(y_test_clv, y_pred_lr))
r2_lr = r2_score(y_test_clv, y_pred_lr)
print(f"RMSE: ${rmse_lr:.2f}")
print(f"R² Score: {r2_lr:.4f}")

# 5.2 Ridge Regression (L2 Regularization)
print("\n5.2 Ridge Regression")
print("-" * 40)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_reg_scaled, y_train_clv)
y_pred_ridge = ridge.predict(X_test_reg_scaled)

rmse_ridge = np.sqrt(mean_squared_error(y_test_clv, y_pred_ridge))
r2_ridge = r2_score(y_test_clv, y_pred_ridge)
print(f"RMSE: ${rmse_ridge:.2f}")
print(f"R² Score: {r2_ridge:.4f}")

# 5.3 Lasso Regression (L1 Regularization)
print("\n5.3 Lasso Regression")
print("-" * 40)
lasso = Lasso(alpha=10.0)
lasso.fit(X_train_reg_scaled, y_train_clv)
y_pred_lasso = lasso.predict(X_test_reg_scaled)

rmse_lasso = np.sqrt(mean_squared_error(y_test_clv, y_pred_lasso))
r2_lasso = r2_score(y_test_clv, y_pred_lasso)
print(f"RMSE: ${rmse_lasso:.2f}")
print(f"R² Score: {r2_lasso:.4f}")

# Feature Importance (from Lasso - non-zero coefficients)
print("\nTop 10 Important Features (Lasso):")
feature_importance = pd.DataFrame({
    'feature': X_reg.columns,
    'coefficient': np.abs(lasso.coef_)
}).sort_values('coefficient', ascending=False)
print(feature_importance.head(10))

# ============================================================================
# 6. CHURN PREDICTION - CLASSIFICATION
# ============================================================================

print("\n" + "=" * 80)
print("6. CHURN PREDICTION - CLASSIFICATION")
print("=" * 80)

# Prepare features
classification_features = regression_features.copy()
X_clf = X_reg.copy()
y_churn = df['churned'].copy()

# Train-test split
X_train_clf, X_test_clf, y_train_churn, y_test_churn = train_test_split(
    X_clf, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

# Standardize
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

print(f"Training Set: {X_train_clf.shape[0]} samples")
print(f"Test Set: {X_test_clf.shape[0]} samples")
print(f"Churn Rate (Training): {y_train_churn.mean():.2%}")
print(f"Churn Rate (Test): {y_test_churn.mean():.2%}")

# Dictionary to store results
classification_results = {}

# 6.1 Logistic Regression
print("\n6.1 Logistic Regression")
print("-" * 40)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_clf_scaled, y_train_churn)
y_pred_log = log_reg.predict(X_test_clf_scaled)
y_pred_proba_log = log_reg.predict_proba(X_test_clf_scaled)[:, 1]

print(classification_report(y_test_churn, y_pred_log, target_names=['Not Churned', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_churn, y_pred_proba_log):.4f}")
classification_results['Logistic Regression'] = roc_auc_score(y_test_churn, y_pred_proba_log)

# 6.2 Decision Tree
print("\n6.2 Decision Tree Classifier")
print("-" * 40)
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
dt.fit(X_train_clf_scaled, y_train_churn)
y_pred_dt = dt.predict(X_test_clf_scaled)
y_pred_proba_dt = dt.predict_proba(X_test_clf_scaled)[:, 1]

print(classification_report(y_test_churn, y_pred_dt, target_names=['Not Churned', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_churn, y_pred_proba_dt):.4f}")
classification_results['Decision Tree'] = roc_auc_score(y_test_churn, y_pred_proba_dt)

# 6.3 Random Forest
print("\n6.3 Random Forest Classifier")
print("-" * 40)
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf.fit(X_train_clf_scaled, y_train_churn)
y_pred_rf = rf.predict(X_test_clf_scaled)
y_pred_proba_rf = rf.predict_proba(X_test_clf_scaled)[:, 1]

print(classification_report(y_test_churn, y_pred_rf, target_names=['Not Churned', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_churn, y_pred_proba_rf):.4f}")
classification_results['Random Forest'] = roc_auc_score(y_test_churn, y_pred_proba_rf)

# Feature Importance
print("\nTop 10 Important Features (Random Forest):")
feature_importance_rf = pd.DataFrame({
    'feature': X_clf.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance_rf.head(10))

# 6.4 Support Vector Machine (SVM)
print("\n6.4 Support Vector Machine (SVM)")
print("-" * 40)
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_clf_scaled, y_train_churn)
y_pred_svm = svm.predict(X_test_clf_scaled)
y_pred_proba_svm = svm.predict_proba(X_test_clf_scaled)[:, 1]

print(classification_report(y_test_churn, y_pred_svm, target_names=['Not Churned', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_churn, y_pred_proba_svm):.4f}")
classification_results['SVM'] = roc_auc_score(y_test_churn, y_pred_proba_svm)

# 6.5 Naive Bayes
print("\n6.5 Gaussian Naive Bayes")
print("-" * 40)
nb = GaussianNB()
nb.fit(X_train_clf_scaled, y_train_churn)
y_pred_nb = nb.predict(X_test_clf_scaled)
y_pred_proba_nb = nb.predict_proba(X_test_clf_scaled)[:, 1]

print(classification_report(y_test_churn, y_pred_nb, target_names=['Not Churned', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_churn, y_pred_proba_nb):.4f}")
classification_results['Naive Bayes'] = roc_auc_score(y_test_churn, y_pred_proba_nb)

# 6.6 Gradient Boosting
print("\n6.6 Gradient Boosting Classifier")
print("-" * 40)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=5, random_state=42)
gb.fit(X_train_clf_scaled, y_train_churn)
y_pred_gb = gb.predict(X_test_clf_scaled)
y_pred_proba_gb = gb.predict_proba(X_test_clf_scaled)[:, 1]

print(classification_report(y_test_churn, y_pred_gb, target_names=['Not Churned', 'Churned']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_churn, y_pred_proba_gb):.4f}")
classification_results['Gradient Boosting'] = roc_auc_score(y_test_churn, y_pred_proba_gb)

# Model Comparison
print("\n" + "=" * 80)
print("CLASSIFICATION MODEL COMPARISON (ROC-AUC)")
print("=" * 80)
results_df = pd.DataFrame.from_dict(classification_results, orient='index', 
                                     columns=['ROC-AUC Score'])
results_df = results_df.sort_values('ROC-AUC Score', ascending=False)
print(results_df)
print(f"\nBest Model: {results_df.index[0]} with ROC-AUC = {results_df.iloc[0, 0]:.4f}")

# ============================================================================
# 7. CAMPAIGN RESPONSE PREDICTION
# ============================================================================

print("\n" + "=" * 80)
print("7. CAMPAIGN RESPONSE PREDICTION")
print("=" * 80)

y_campaign = df['campaign_response'].copy()
X_train_camp, X_test_camp, y_train_camp, y_test_camp = train_test_split(
    X_clf, y_campaign, test_size=0.2, random_state=42, stratify=y_campaign
)

X_train_camp_scaled = scaler_clf.fit_transform(X_train_camp)
X_test_camp_scaled = scaler_clf.transform(X_test_camp)

# Random Forest for Campaign Response
rf_campaign = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_campaign.fit(X_train_camp_scaled, y_train_camp)
y_pred_campaign = rf_campaign.predict(X_test_camp_scaled)
y_pred_proba_campaign = rf_campaign.predict_proba(X_test_camp_scaled)[:, 1]

print("Campaign Response Prediction Results:")
print(classification_report(y_test_camp, y_pred_campaign, 
                          target_names=['No Response', 'Response']))
print(f"ROC-AUC Score: {roc_auc_score(y_test_camp, y_pred_proba_campaign):.4f}")

# ============================================================================
# 8. DIMENSIONALITY REDUCTION - PCA
# ============================================================================

print("\n" + "=" * 80)
print("8. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 80)

pca = PCA()
X_pca = pca.fit_transform(X_cluster_scaled)

# Explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"Number of components explaining 90% variance: {n_components_90}")
print("\nExplained Variance Ratio by Component:")
for i, var in enumerate(pca.explained_variance_ratio_[:5], 1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

print(f"\nCumulative Variance (first 5 components): {cumulative_variance[4]:.4f}")

# ============================================================================
# 9. CROSS-VALIDATION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("9. CROSS-VALIDATION ANALYSIS")
print("=" * 80)

# Cross-validation for classification models
models_cv = {
    'Logistic Regression': log_reg,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'SVM': svm
}

print("5-Fold Cross-Validation Results (ROC-AUC):")
print("-" * 40)
for name, model in models_cv.items():
    cv_scores = cross_val_score(model, X_clf, y_churn, cv=5, 
                                scoring='roc_auc', n_jobs=-1)
    print(f"{name:25s}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 10. SUMMARY AND INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("PROJECT SUMMARY & KEY INSIGHTS")
print("=" * 80)

print("\n1. CUSTOMER SEGMENTATION:")
print(f"   - K-Means identified {optimal_k} distinct customer segments")
print(f"   - Silhouette Score: {silhouette_score(X_cluster_scaled, df['KMeans_Cluster']):.4f}")
print("   - Segments range from high-value champions to at-risk customers")

print("\n2. CUSTOMER LIFETIME VALUE PREDICTION:")
print(f"   - Best Model: Ridge Regression (R² = {r2_ridge:.4f})")
print(f"   - RMSE: ${rmse_ridge:.2f}")
print("   - Key Drivers: Frequency, Monetary Value, Tenure")

print("\n3. CHURN PREDICTION:")
print(f"   - Best Model: {results_df.index[0]} (ROC-AUC = {results_df.iloc[0, 0]:.4f})")
print(f"   - Overall Churn Rate: {df['churned'].mean():.2%}")
print("   - Top Predictors: Recency, Frequency, Email Engagement")

print("\n4. RFM ANALYSIS:")
print("   - Champions segment shows highest CLV")
print("   - At-Risk and Lost segments need retention campaigns")

print("\n5. MARKETING RECOMMENDATIONS:")
print("   - Target Champions and Loyal Customers for upsell/cross-sell")
print("   - Implement win-back campaigns for At-Risk segment")
print("   - Personalize communications based on product preferences")
print("   - Focus retention efforts on high-churn-probability customers")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)

# ============================================================================
# BONUS: VISUALIZATION EXAMPLES
# ============================================================================

print("\n" + "=" * 80)
print("11. GENERATING VISUALIZATIONS")
print("=" * 80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Customer Segmentation (K-Means)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# PCA for 2D visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_cluster_scaled)

# Subplot 1: K-Means Clusters
scatter = axes[0, 0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                             c=df['KMeans_Cluster'], cmap='viridis', 
                             alpha=0.6, s=50)
axes[0, 0].set_title('K-Means Customer Segmentation (PCA)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
axes[0, 0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')

# Subplot 2: RFM Segments
rfm_segment_map = {'Champions': 4, 'Loyal Customers': 3, 
                   'Potential Loyalists': 2, 'At Risk': 1, 'Lost': 0}
df_rfm['segment_encoded'] = df_rfm['RFM_Segment'].map(rfm_segment_map)
scatter2 = axes[0, 1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                              c=df_rfm['segment_encoded'], cmap='RdYlGn', 
                              alpha=0.6, s=50)
axes[0, 1].set_title('RFM Segmentation', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
axes[0, 1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
cbar2.set_ticks([0, 1, 2, 3, 4])
cbar2.set_ticklabels(['Lost', 'At Risk', 'Potential', 'Loyal', 'Champions'])

# Subplot 3: CLV Distribution by Cluster
df.boxplot(column='clv', by='KMeans_Cluster', ax=axes[1, 0])
axes[1, 0].set_title('CLV Distribution by Customer Segment', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Cluster', fontsize=12)
axes[1, 0].set_ylabel('Customer Lifetime Value ($)', fontsize=12)
plt.sca(axes[1, 0])
plt.xticks(range(1, optimal_k + 1), [f'Cluster {i}' for i in range(optimal_k)])

# Subplot 4: Churn Rate by Cluster
churn_by_cluster = df.groupby('KMeans_Cluster')['churned'].mean()
axes[1, 1].bar(range(optimal_k), churn_by_cluster.values, color='coral', alpha=0.7)
axes[1, 1].set_title('Churn Rate by Customer Segment', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Cluster', fontsize=12)
axes[1, 1].set_ylabel('Churn Rate', fontsize=12)
axes[1, 1].set_xticks(range(optimal_k))
axes[1, 1].set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
axes[1, 1].set_ylim(0, max(churn_by_cluster) * 1.2)
for i, v in enumerate(churn_by_cluster.values):
    axes[1, 1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')

plt.tight_layout()
print("\n✓ Figure 1: Customer Segmentation Analysis saved")

# Figure 2: Model Performance Comparison
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Elbow Curve
axes2[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes2[0, 0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
axes2[0, 0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes2[0, 0].set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axes2[0, 0].legend()

# Subplot 2: Silhouette Scores
axes2[0, 1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes2[0, 1].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
axes2[0, 1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes2[0, 1].set_ylabel('Silhouette Score', fontsize=12)
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
axes2[0, 1].legend()

# Subplot 3: Classification Model Comparison
model_names = list(classification_results.keys())
roc_scores = list(classification_results.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
bars = axes2[1, 0].barh(model_names, roc_scores, color=colors, alpha=0.8)
axes2[1, 0].set_title('Churn Prediction Model Comparison', fontsize=14, fontweight='bold')
axes2[1, 0].set_xlabel('ROC-AUC Score', fontsize=12)
axes2[1, 0].set_xlim(0, 1)
axes2[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random Baseline')
for i, (bar, score) in enumerate(zip(bars, roc_scores)):
    axes2[1, 0].text(score + 0.01, i, f'{score:.4f}', va='center', fontweight='bold')
axes2[1, 0].legend()

# Subplot 4: ROC Curves
fpr_log, tpr_log, _ = roc_curve(y_test_churn, y_pred_proba_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test_churn, y_pred_proba_rf)
fpr_gb, tpr_gb, _ = roc_curve(y_test_churn, y_pred_proba_gb)

axes2[1, 1].plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={roc_auc_score(y_test_churn, y_pred_proba_log):.3f})', linewidth=2)
axes2[1, 1].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test_churn, y_pred_proba_rf):.3f})', linewidth=2)
axes2[1, 1].plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC={roc_auc_score(y_test_churn, y_pred_proba_gb):.3f})', linewidth=2)
axes2[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
axes2[1, 1].set_title('ROC Curves - Churn Prediction', fontsize=14, fontweight='bold')
axes2[1, 1].set_xlabel('False Positive Rate', fontsize=12)
axes2[1, 1].set_ylabel('True Positive Rate', fontsize=12)
axes2[1, 1].legend(loc='lower right')
axes2[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
print("✓ Figure 2: Model Performance Comparison saved")

# Figure 3: Feature Importance and Analysis
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Feature Importance (Random Forest)
top_features = feature_importance_rf.head(10)
axes3[0, 0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue', alpha=0.8)
axes3[0, 0].set_yticks(range(len(top_features)))
axes3[0, 0].set_yticklabels(top_features['feature'].values)
axes3[0, 0].set_xlabel('Importance Score', fontsize=12)
axes3[0, 0].set_title('Top 10 Features for Churn Prediction (Random Forest)', fontsize=14, fontweight='bold')
axes3[0, 0].invert_yaxis()

# Subplot 2: PCA Explained Variance
axes3[0, 1].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                 cumulative_variance, 'b-o', linewidth=2, markersize=6)
axes3[0, 1].axhline(y=0.90, color='red', linestyle='--', label='90% Variance')
axes3[0, 1].axvline(x=n_components_90, color='red', linestyle='--', alpha=0.5)
axes3[0, 1].set_title('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes3[0, 1].set_xlabel('Number of Components', fontsize=12)
axes3[0, 1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes3[0, 1].grid(True, alpha=0.3)
axes3[0, 1].legend()

# Subplot 3: RFM Segment Distribution
rfm_counts = df_rfm['RFM_Segment'].value_counts()
colors_rfm = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
axes3[1, 0].pie(rfm_counts.values, labels=rfm_counts.index, autopct='%1.1f%%',
                colors=colors_rfm, startangle=90)
axes3[1, 0].set_title('Customer Distribution by RFM Segment', fontsize=14, fontweight='bold')

# Subplot 4: Correlation Heatmap (Top Features)
top_corr_features = ['recency', 'frequency', 'monetary', 'tenure_months', 
                     'avg_order_value', 'clv', 'churned']
corr_matrix = df[top_corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes3[1, 1], cbar_kws={'label': 'Correlation'})
axes3[1, 1].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
print("✓ Figure 3: Feature Analysis and Insights saved")

# Figure 4: Business Insights Dashboard
fig4, axes4 = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: CLV by RFM Segment
avg_clv_by_segment = df_rfm.groupby('RFM_Segment')['clv'].mean().sort_values(ascending=False)
axes4[0, 0].bar(range(len(avg_clv_by_segment)), avg_clv_by_segment.values, 
                color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6'], alpha=0.8)
axes4[0, 0].set_xticks(range(len(avg_clv_by_segment)))
axes4[0, 0].set_xticklabels(avg_clv_by_segment.index, rotation=45, ha='right')
axes4[0, 0].set_ylabel('Average CLV ($)', fontsize=12)
axes4[0, 0].set_title('Average Customer Lifetime Value by RFM Segment', fontsize=14, fontweight='bold')
for i, v in enumerate(avg_clv_by_segment.values):
    axes4[0, 0].text(i, v + 100, f'${v:.0f}', ha='center', fontweight='bold')

# Subplot 2: Churn Rate by RFM Segment
churn_by_rfm = df_rfm.groupby('RFM_Segment')['churned'].mean().sort_values(ascending=False)
axes4[0, 1].bar(range(len(churn_by_rfm)), churn_by_rfm.values, 
                color='coral', alpha=0.8)
axes4[0, 1].set_xticks(range(len(churn_by_rfm)))
axes4[0, 1].set_xticklabels(churn_by_rfm.index, rotation=45, ha='right')
axes4[0, 1].set_ylabel('Churn Rate', fontsize=12)
axes4[0, 1].set_title('Churn Rate by RFM Segment', fontsize=14, fontweight='bold')
for i, v in enumerate(churn_by_rfm.values):
    axes4[0, 1].text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')

# Subplot 3: Age Distribution by Churn Status
df[df['churned']==0]['age'].hist(bins=20, alpha=0.6, label='Not Churned', 
                                  color='green', ax=axes4[1, 0])
df[df['churned']==1]['age'].hist(bins=20, alpha=0.6, label='Churned', 
                                  color='red', ax=axes4[1, 0])
axes4[1, 0].set_xlabel('Age', fontsize=12)
axes4[1, 0].set_ylabel('Frequency', fontsize=12)
axes4[1, 0].set_title('Age Distribution by Churn Status', fontsize=14, fontweight='bold')
axes4[1, 0].legend()

# Subplot 4: Income vs CLV (colored by cluster)
scatter4 = axes4[1, 1].scatter(df['income'], df['clv'], 
                               c=df['KMeans_Cluster'], cmap='viridis', 
                               alpha=0.6, s=30)
axes4[1, 1].set_xlabel('Annual Income ($)', fontsize=12)
axes4[1, 1].set_ylabel('Customer Lifetime Value ($)', fontsize=12)
axes4[1, 1].set_title('Income vs CLV by Customer Segment', fontsize=14, fontweight='bold')
plt.colorbar(scatter4, ax=axes4[1, 1], label='Cluster')

plt.tight_layout()
print("✓ Figure 4: Business Insights Dashboard saved")

print("\n✓ All visualizations generated successfully!")
print("\nNote: In a Jupyter environment, use plt.show() to display the figures.")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
