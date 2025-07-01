import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataProcessor:
    def __init__(self, raw_data_path):
       
        self.raw_data_path = raw_data_path
        self.df = None
        self.rfm_df = None

    def load_data(self):
       
        try:
            self.df = pd.read_csv(self.raw_data_path)
            print(f"Data loaded successfully from {self.raw_data_path}. Shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: File not found at {self.raw_data_path}")
            return False
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return False

    def preprocess_data(self):
        if self.df is None:
            print("Error: DataFrame not loaded. Call load_data() first.")
            return False
        print("Starting data preprocessing...")
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
        self.df['transaction_date'] = self.df['TransactionStartTime'].dt.date
        print("TransactionStartTime converted and date extracted.")
        return True

    def calculate_rfm(self):
        if self.df is None:
            print("Error: DataFrame not loaded. Call load_data() first.")
            return False
        print("Calculating RFM features...")
        # Get the latest transaction date in the entire dataset
        max_transaction_date = self.df['TransactionStartTime'].max()

        # Calculate Recency, Frequency, Monetary at CustomerId level
        self.rfm_df = self.df.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda date: (max_transaction_date - date.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum')
        ).reset_index()

        print("RFM features calculated.")
        return True

    def define_default_proxy(self):
        if self.df is None or self.rfm_df is None:
            print("Error: DataFrames not loaded or RFM not calculated. Call load_data() and calculate_rfm() first.")
            return False
        print("Defining proxy default variable...")
        # Identify customers with at least one fraudulent transaction
        fraud_customers = self.df[self.df['FraudResult'] == 1]['CustomerId'].unique()

        # Create a 'Default' column in rfm_df
        self.rfm_df['Default'] = self.rfm_df['CustomerId'].isin(fraud_customers).astype(int)
        print("Proxy default variable defined.")
        return True

    def remove_outliers_zscore(self, columns=None, threshold=3):
        if self.df is None:
            print("Error: DataFrame not loaded. Call load_data() first.")
            return False
        
        print(f"Removing outliers with Z-score > {threshold}...")
        
        # If no columns specified, use all numerical columns
        if columns is None:
            columns = self.df.select_dtypes(include=np.number).columns.tolist()
        
        initial_shape = self.df.shape
        outlier_mask = pd.Series([False] * len(self.df), index=self.df.index)
        
        for col in columns:
            if col in self.df.columns:
                # Calculate Z-scores
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                col_outliers = z_scores > threshold
                outlier_mask = outlier_mask | col_outliers
                
                outlier_count = col_outliers.sum()
                if outlier_count > 0:
                    print(f"  - {col}: {outlier_count} outliers removed")
        
        # Remove outliers
        self.df = self.df[~outlier_mask]
        
        final_shape = self.df.shape
        removed_count = initial_shape[0] - final_shape[0]
        
        print(f"Outlier removal completed:")
        print(f"  - Initial rows: {initial_shape[0]}")
        print(f"  - Final rows: {final_shape[0]}")
        print(f"  - Rows removed: {removed_count}")
        print(f"  - Percentage removed: {(removed_count/initial_shape[0])*100:.2f}%")
        
        return True

    def clean_and_save_data(self, output_filename="cleaned-data.csv"):
        if self.df is None:
            print("Error: DataFrame not loaded. Call load_data() first.")
            return False
        
        print("=== Data Cleaning and Saving Pipeline ===")
        
        # Step 1: Remove outliers from key numerical columns
        print("\n1. Removing outliers (Z-score > 3)...")
        if not self.remove_outliers_zscore(columns=['Amount', 'Value']):
            print("Failed to remove outliers")
            return False
        
        # Step 2: Save cleaned data
        import os
        from pathlib import Path
        
        # Create processed directory if it doesn't exist
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = processed_dir / output_filename
        
        try:
            print(f"\n2. Saving cleaned data to {output_path}...")
            self.df.to_csv(output_path, index=False)
            print(f"✅ Successfully saved cleaned data!")
            print(f"📊 Final dataset shape: {self.df.shape}")
            print(f"💾 File saved at: {output_path}")
            
            # Display some statistics
            print(f"\n📈 Dataset Summary:")
            print(f"   - Total rows: {len(self.df)}")
            print(f"   - Total columns: {len(self.df.columns)}")
            if 'FraudResult' in self.df.columns:
                fraud_count = self.df['FraudResult'].sum()
                fraud_rate = self.df['FraudResult'].mean()
                print(f"   - Fraud cases: {fraud_count} ({fraud_rate:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False

    def run_pipeline(self, processed_data_output_path):
        if not self.load_data():
            return None
        if not self.preprocess_data():
            return None
        if not self.calculate_rfm():
            return None
        if not self.define_default_proxy():
            return None

        # Save processed data
        try:
            self.rfm_df.to_csv(processed_data_output_path, index=False)
            print(f"Processed customer data saved to {processed_data_output_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")

        return self.rfm_df

class EDAReportGenerator:
    def __init__(self, df, output_dir="eda_plots"):
        self.df = df
        if self.df is None:
            raise ValueError("DataFrame cannot be None for EDAReportGenerator.")
        
        # Create output directory for plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set figure size for all plots
        plt.rcParams['figure.figsize'] = (12, 8)
    def get_data_overview(self):
        report = "### 1. Overview of the Data\n"
        report += f"  - Number of Rows: {self.df.shape[0]}\n"
        report += f"  - Number of Columns: {self.df.shape[1]}\n"
        report += "  - Data Types:\n"
        report += self.df.dtypes.to_string() + "\n"
        return report   
    def get_summary_statistics(self):
        report = "\n### 2. Summary Statistics (Numerical Features)\n"
        report += self.df.describe().to_string() + "\n"
        return report
    def numerical_distribution(self):
        report = "\n### 3. Distribution of Numerical Features\n"
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            report += f"  - **{col}**:\n"
            report += f"    - Mean: {self.df[col].mean():.2f}\n"
            report += f"    - Median: {self.df[col].median():.2f}\n"
            report += f"    - Skewness: {self.df[col].skew():.2f} (Positive: Right-skewed, Negative: Left-skewed, ~0: Symmetric)\n"
            report += f"    - Std Dev: {self.df[col].std():.2f}\n"
            if col in ['Amount', 'Value']:
                report += f"    - Range: {self.df[col].min():.2f} to {self.df[col].max():.2f}\n"
                report += "    - Observations: Typically highly skewed with many small transactions and a few very large ones. 'Amount' includes negative values (debits/credits).\n"
                report += f"    - 📊 **Visualization**: See `{col}_distribution_analysis.png` for detailed distribution plots.\n"
            elif col == 'FraudResult':
                report += f"    - Observations: Highly imbalanced (mean {self.df[col].mean():.4f}), indicating rare occurrences of fraud.\n"
                report += f"    - 📊 **Visualization**: See `fraud_result_analysis.png` for fraud distribution analysis.\n"
        return report

    def numerical_distribution_plots(self):
       
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        
        for col in numerical_cols:
            if col in ['Amount', 'Value']:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Distribution Analysis: {col}', fontsize=16, fontweight='bold')
                
                # Histogram
                axes[0, 0].hist(self.df[col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[0, 0].set_title(f'{col} - Histogram')
                axes[0, 0].set_xlabel(col)
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Box plot
                axes[0, 1].boxplot(self.df[col], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
                axes[0, 1].set_title(f'{col} - Box Plot')
                axes[0, 1].set_ylabel(col)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Q-Q plot
                from scipy import stats
                stats.probplot(self.df[col], dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title(f'{col} - Q-Q Plot (Normal Distribution)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Log-transformed histogram (for highly skewed data)
                if self.df[col].min() > 0:  # Only if all values are positive
                    log_data = np.log1p(self.df[col])
                    axes[1, 1].hist(log_data, bins=50, alpha=0.7, color='salmon', edgecolor='black')
                    axes[1, 1].set_title(f'{col} - Log-Transformed Histogram')
                    axes[1, 1].set_xlabel(f'log({col})')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    # If negative values exist, show cumulative distribution
                    axes[1, 1].hist(self.df[col], bins=50, alpha=0.7, color='salmon', 
                                   edgecolor='black', cumulative=True, density=True)
                    axes[1, 1].set_title(f'{col} - Cumulative Distribution')
                    axes[1, 1].set_xlabel(col)
                    axes[1, 1].set_ylabel('Cumulative Probability')
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            
                
            elif col == 'FraudResult':
                # Special handling for binary target variable
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f'Fraud Result Analysis', fontsize=16, fontweight='bold')
                
                # Bar plot
                fraud_counts = self.df[col].value_counts()
                axes[0].bar(fraud_counts.index, fraud_counts.values, 
                           color=['lightblue', 'lightcoral'], alpha=0.8)
                axes[0].set_title('Fraud Result Distribution')
                axes[0].set_xlabel('Fraud Result')
                axes[0].set_ylabel('Count')
                axes[0].set_xticks([0, 1])
                axes[0].set_xticklabels(['No Fraud (0)', 'Fraud (1)'])
                
                # Add count labels on bars
                for i, v in enumerate(fraud_counts.values):
                    axes[0].text(i, v + max(fraud_counts.values) * 0.01, 
                                str(v), ha='center', va='bottom', fontweight='bold')
                
                # Pie chart
                axes[1].pie(fraud_counts.values, labels=['No Fraud', 'Fraud'], 
                           autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
                axes[1].set_title('Fraud Result Proportion')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'fraud_result_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
    def categorical_distribution(self):
        report = "\n### 4. Distribution of Categorical Features\n"
        categorical_cols = self.df.select_dtypes(include='object').columns
        for col in categorical_cols:
            if col in ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']:
                report += f"  - **{col}**: {self.df[col].nunique()} unique values. Primarily identifiers or datetime strings.\n"
            else:
                report += f"  - **{col}** (Top 5):\n"
                report += self.df[col].value_counts(normalize=True).head(5).mul(100).to_string() + "%\n"
                if col in ['CurrencyCode', 'CountryCode']:
                    report += "    - Observations: Highly concentrated, e.g., mostly UGX and Country Code 256.\n"
                elif col == 'ProductCategory':
                    report += "    - Observations: Dominant categories include 'airtime' and 'financial_services'.\n"
                elif col == 'ChannelId':
                    report += "    - Observations: 'ChannelId_3' and 'ChannelId_2' are most frequent; 'ChannelId_5' (pay later) is particularly relevant.\n"
                report += f"    - 📊 **Visualization**: See `{col}_distribution.png` for detailed category analysis.\n"
        return report
    def categorical_distribution_plots(self):
        categorical_cols = self.df.select_dtypes(include='object').columns
        
        for col in categorical_cols:
            if col not in ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime']:
                # Get top 10 categories for better visualization
                top_categories = self.df[col].value_counts().head(10)
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                fig.suptitle(f'{col} Distribution Analysis', fontsize=16, fontweight='bold')
                
                # Bar plot
                bars = axes[0].bar(range(len(top_categories)), top_categories.values, 
                                  color=plt.cm.Set3(np.linspace(0, 1, len(top_categories))))
                axes[0].set_title(f'{col} - Top 10 Categories')
                axes[0].set_xlabel('Categories')
                axes[0].set_ylabel('Count')
                axes[0].set_xticks(range(len(top_categories)))
                axes[0].set_xticklabels(top_categories.index, rotation=45, ha='right')
                axes[0].grid(True, alpha=0.3)
                
                # Add count labels on bars
                for i, v in enumerate(top_categories.values):
                    axes[0].text(i, v + max(top_categories.values) * 0.01, 
                                str(v), ha='center', va='bottom', fontsize=8)
                
                # Pie chart (top 5 categories)
                top_5 = top_categories.head(5)
                axes[1].pie(top_5.values, labels=top_5.index, autopct='%1.1f%%', 
                           startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(top_5))))
                axes[1].set_title(f'{col} - Top 5 Categories Proportion')
                
                plt.tight_layout()
                plt.show()
    def correlation_analysis(self):
        report = "\n### 5. Correlation Analysis (Numerical Features)\n"
        numerical_df = self.df.select_dtypes(include=np.number)
        if not numerical_df.empty:
            correlation_matrix = numerical_df.corr()
            report += "  - Correlation Matrix:\n"
            report += correlation_matrix.to_string() + "\n"
            report += "  - Key Observations:\n"
            if 'Amount' in correlation_matrix.columns and 'Value' in correlation_matrix.columns:
                report += "    - Strong positive correlation between 'Amount' (absolute magnitude) and 'Value'.\n"
            if 'FraudResult' in correlation_matrix.columns:
                report += "    - 'FraudResult' correlations with other numerical features might be weak due to its imbalance, but large magnitudes in 'Amount'/'Value' could be indicative.\n"
            report += "    - 📊 **Visualization**: See `correlation_heatmap.png` for correlation heatmap.\n"
        else:
            report += "  - No numerical features found for correlation analysis.\n"
        return report
    def correlation_heatmap(self):
        numerical_df = self.df.select_dtypes(include=np.number)
        
        if not numerical_df.empty and numerical_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            
            # Calculate correlation matrix
            correlation_matrix = numerical_df.corr()
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            
            plt.title('Correlation Heatmap - Numerical Features', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
 
    def detect_outliers(self):
        report = "\n### 7. Outlier Detection (Numerical Features) - Z-Score Method\n"
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            if col in ['Amount', 'Value']: # Focus on key numerical features for outliers
                # Calculate Z-scores
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                
                # Define outliers as observations with |Z-score| > 3 (standard threshold)
                outliers_mask = z_scores > 3
                outliers_count = outliers_mask.sum()
                outliers_percentage = (outliers_count / len(self.df)) * 100
                
                # Get some statistics about the outliers
                outlier_values = self.df[col][outliers_mask]
                
                report += f"  - **{col}**:\n"
                report += f"    - Mean: {self.df[col].mean():.2f}\n"
                report += f"    - Standard Deviation: {self.df[col].std():.2f}\n"
                report += f"    - Number of Outliers (|Z-score| > 3): {outliers_count}\n"
                report += f"    - Percentage of Outliers: {outliers_percentage:.2f}%\n"
                
                if outliers_count > 0:
                    report += f"    - Outlier Range: {outlier_values.min():.2f} to {outlier_values.max():.2f}\n"
                    report += f"    - Max Z-score: {z_scores.max():.2f}\n"
                
                report += "    - Observations: Z-score method identifies extreme values that deviate significantly from the mean. "
                report += "High Z-scores may indicate potential fraud, data entry errors, or legitimate large transactions.\n"
                report += f"    - 📊 **Visualization**: See `{col}_outlier_analysis.png` for detailed outlier analysis.\n"
                
            elif col == 'FraudResult':
                report += f"  - **{col}**: As a binary flag, 'FraudResult' doesn't have numerical outliers in the traditional sense, "
                report += "but the rare occurrences of '1' (fraud) represent an outlier class that needs special handling.\n"
        return report
    def outlier_analysis_plots(self):
        numerical_cols = self.df.select_dtypes(include=np.number).columns
        
        for col in numerical_cols:
            if col in ['Amount', 'Value']:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Outlier Analysis: {col}', fontsize=16, fontweight='bold')
                
                # Box plot
                axes[0, 0].boxplot(self.df[col], patch_artist=True, boxprops=dict(facecolor='lightblue'))
                axes[0, 0].set_title(f'{col} - Box Plot')
                axes[0, 0].set_ylabel(col)
                axes[0, 0].grid(True, alpha=0.3)
                
                # Z-score distribution
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                axes[0, 1].hist(z_scores, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[0, 1].axvline(x=3, color='red', linestyle='--', label='Z-score = 3 (Outlier threshold)')
                axes[0, 1].set_title(f'{col} - Z-Score Distribution')
                axes[0, 1].set_xlabel('Absolute Z-Score')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # Scatter plot with outlier highlighting
                outliers_mask = z_scores > 3
                axes[1, 0].scatter(range(len(self.df)), self.df[col], 
                                  c=outliers_mask, cmap='viridis', alpha=0.6)
                axes[1, 0].set_title(f'{col} - Values with Outlier Highlighting')
                axes[1, 0].set_xlabel('Index')
                axes[1, 0].set_ylabel(col)
                axes[1, 0].grid(True, alpha=0.3)
                
                # IQR method comparison
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                
                axes[1, 1].scatter(range(len(self.df)), self.df[col], 
                                  c=iqr_outliers, cmap='plasma', alpha=0.6)
                axes[1, 1].axhline(y=lower_bound, color='red', linestyle='--', label='IQR Lower Bound')
                axes[1, 1].axhline(y=upper_bound, color='red', linestyle='--', label='IQR Upper Bound')
                axes[1, 1].set_title(f'{col} - IQR Outlier Detection')
                axes[1, 1].set_xlabel('Index')
                axes[1, 1].set_ylabel(col)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()

    def identify_missing_values(self):
        report = "\n### 6. Identifying Missing Values\n"
        missing_values = self.df.isnull().sum()
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage': missing_percentage})
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if not missing_df.empty:
            report += "  - Columns with Missing Values:\n"
            report += missing_df.to_string() + "\n"
            report += "  - Observations: Missing values will require appropriate imputation strategies (e.g., mean, median, mode, or more sophisticated methods).\n"
            report += "    - 📊 **Visualization**: See `missing_values_analysis.png` for missing values analysis.\n"
        else:
            report += "  - No missing values found in the dataset.\n"
        return report
    def missing_values_plot(self):
        """Create missing values visualization."""
        missing_values = self.df.isnull().sum()
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        
        if missing_values.sum() > 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
            
            # Bar plot of missing values
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Count': missing_values.values,
                'Missing Percentage': missing_percentage.values
            }).sort_values('Missing Count', ascending=False)
            
            axes[0].bar(range(len(missing_df)), missing_df['Missing Count'], 
                       color='lightcoral', alpha=0.8)
            axes[0].set_title('Missing Values Count by Column')
            axes[0].set_xlabel('Columns')
            axes[0].set_ylabel('Missing Count')
            axes[0].set_xticks(range(len(missing_df)))
            axes[0].set_xticklabels(missing_df['Column'], rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # Percentage bar plot
            axes[1].bar(range(len(missing_df)), missing_df['Missing Percentage'], 
                       color='lightblue', alpha=0.8)
            axes[1].set_title('Missing Values Percentage by Column')
            axes[1].set_xlabel('Columns')
            axes[1].set_ylabel('Missing Percentage (%)')
            axes[1].set_xticks(range(len(missing_df)))
            axes[1].set_xticklabels(missing_df['Column'], rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

    def fraud_analysis_plots(self):
        if 'FraudResult' in self.df.columns:
            # Fraud vs Amount/Value analysis
            numerical_cols = ['Amount', 'Value']
            available_cols = [col for col in numerical_cols if col in self.df.columns]
            
            if available_cols:
                fig, axes = plt.subplots(1, len(available_cols), figsize=(15, 6))
                if len(available_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(available_cols):
                    # Box plot by fraud result
                    fraud_data = [self.df[self.df['FraudResult'] == 0][col], 
                                 self.df[self.df['FraudResult'] == 1][col]]
                    axes[i].boxplot(fraud_data, labels=['No Fraud', 'Fraud'], 
                                   patch_artist=True, 
                                   boxprops=dict(facecolor='lightblue'),
                                   medianprops=dict(color='red'))
                    axes[i].set_title(f'{col} Distribution by Fraud Result')
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'fraud_amount_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Violin plot for better distribution visualization
                fig, axes = plt.subplots(1, len(available_cols), figsize=(15, 6))
                if len(available_cols) == 1:
                    axes = [axes]
                
                for i, col in enumerate(available_cols):
                    sns.violinplot(data=self.df, x='FraudResult', y=col, ax=axes[i])
                    axes[i].set_title(f'{col} Distribution by Fraud Result (Violin Plot)')
                    axes[i].set_xlabel('Fraud Result')
                    axes[i].set_ylabel(col)
                    axes[i].set_xticks([0, 1])
                    axes[i].set_xticklabels(['No Fraud', 'Fraud'])
                
                plt.tight_layout()
                plt.show()

    
    

    

 
