# Summary Report: Customer Segmentation

**Problem and data:** An e-commerce marketing team wants to stop sending identical emails to all customers. I used the Online Retail II dataset (UK customers only, 5,350 customers after cleaning) to group customers into segments.

**Features and preprocessing:** I built RFM features per customer (Recency, Frequency, Monetary). I capped outliers using the IQR method, applied log1p to Frequency and Monetary because they were right-skewed, and scaled everything with StandardScaler, since all three algorithms depend on distances.

**Algorithm results:** K-Means (k=4) performed best on all internal metrics (Silhouette 0.374, Davies-Bouldin 0.905, Calinski-Harabasz 5797) and was identical across 5 random seeds. Agglomerative (Ward) was very close. DBSCAN scored worst (Silhouette 0.201) and labelled 4.4% of customers as noise. The metrics matched my business intuition, since K-Means gave clean, balanced segments.

**Segments (K-Means):**
- **Champions:** recent, frequent, highest spend.
- **Loyal Customers:** repeat buyers with solid spend.
- **New Customers:** bought once or twice recently, low spend.
- **Hibernating:** inactive for over a year, low spend.

Hierarchical clustering also found an At-Risk group (formerly good customers, inactive for about 455 days), which K-Means did not separate.

**Next steps:** add features such as product category, returns, discount usage and email engagement; test semi-supervised refinement using marketing feedback; and build a real-time scoring API using `predict_segment()`.