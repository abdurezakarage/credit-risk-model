import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ProxyTargetEngineer:
    def __init__(self, df, transaction_col='TransactionStartTime', customer_col='CustomerId', value_col='Value', transaction_id_col='TransactionId'):
        self.df = df.copy()
        self.transaction_col = transaction_col
        self.customer_col = customer_col
        self.value_col = value_col
        self.transaction_id_col = transaction_id_col
        self.rfm_df = None

    def calculate_rfm(self, snapshot_date=None):
        self.df[self.transaction_col] = pd.to_datetime(self.df[self.transaction_col])
        if snapshot_date is None:
            snapshot_date = self.df[self.transaction_col].max() + pd.Timedelta(days=1)

        rfm = self.df.groupby(self.customer_col).agg(
            Recency=(self.transaction_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(self.transaction_id_col, 'count'),
            Monetary=(self.value_col, 'sum')
        ).reset_index()

        self.rfm_df = rfm
        return rfm

    def perform_clustering(self, n_clusters=3, random_state=42):
        if self.rfm_df is None:
            raise ValueError("Run calculate_rfm() before clustering")

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(self.rfm_df[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

        return self.rfm_df

    def assign_high_risk_label(self):
        cluster_summary = self.rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        high_risk_cluster = cluster_summary.sort_values(['Frequency', 'Monetary'], ascending=[True, True]).index[0]

        self.rfm_df['is_high_risk'] = (self.rfm_df['Cluster'] == high_risk_cluster).astype(int)
        return self.rfm_df[[self.customer_col, 'is_high_risk']]

    def merge_target_with_main(self):
        target_df = self.assign_high_risk_label()
        merged_df = self.df.merge(target_df, on=self.customer_col, how='left')
        merged_df['is_high_risk'].fillna(0, inplace=True)  # In case any customer didn't match (shouldn't happen)
        return merged_df

