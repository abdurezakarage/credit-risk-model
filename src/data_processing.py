import pandas as pd
import numpy as np

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
        """
        Remove outliers based on Z-score method.
        
        Args:
            columns (list): List of numerical columns to check for outliers. 
                          If None, uses all numerical columns.
            threshold (float): Z-score threshold for outlier detection. Default is 3.
        
        Returns:
            bool: True if successful, False otherwise
        """
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
        """
        Clean data by removing outliers (Z-score > 3) and save to CSV file.
        
        Args:
            output_filename (str): Name of the output CSV file. Default is "cleaned-data.csv"
        
        Returns:
            bool: True if successful, False otherwise
        """
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
    def __init__(self, df):
      
        self.df = df
        if self.df is None:
            raise ValueError("DataFrame cannot be None for EDAReportGenerator.")

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

    def analyze_numerical_distribution(self):
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
            elif col == 'FraudResult':
                report += f"    - Observations: Highly imbalanced (mean {self.df[col].mean():.4f}), indicating rare occurrences of fraud.\n"
        return report

    def analyze_categorical_distribution(self):
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
        return report

    def perform_correlation_analysis(self):
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
        else:
            report += "  - No numerical features found for correlation analysis.\n"
        return report

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
        else:
            report += "  - No missing values found in the dataset.\n"
        return report

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
                
            elif col == 'FraudResult':
                report += f"  - **{col}**: As a binary flag, 'FraudResult' doesn't have numerical outliers in the traditional sense, "
                report += "but the rare occurrences of '1' (fraud) represent an outlier class that needs special handling.\n"
        return report

    def generate_eda_report(self):
        full_report = "## Exploratory Data Analysis (EDA) Report\n\n"
        full_report += self.get_data_overview()
        full_report += self.get_summary_statistics()
        full_report += self.analyze_numerical_distribution()
        full_report += self.analyze_categorical_distribution()
        full_report += self.perform_correlation_analysis()
        full_report += self.identify_missing_values()
        full_report += self.detect_outliers()
        return full_report