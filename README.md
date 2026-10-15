# Customer Segmentation using Unsupervised Learning

RFM-based customer segmentation on the Online Retail II dataset, comparing **K-Means**, **Agglomerative Hierarchical Clustering** and **DBSCAN**.


---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Customer Personas](#customer-personas)
6. [Screenshots](#screenshots)
7. [Project Structure](#project-structure)
8. [How to Run](#how-to-run)
9. [Using the Saved Model](#using-the-saved-model)
10. [Key Findings](#key-findings)
11. [Future Work](#future-work)
12. [Video Explanation](#video-explanation)
13. [Author](#author)

---

## Project Overview
An e-commerce marketing team wants to stop sending identical promotional emails to every customer. This project groups customers into meaningful segments so that each segment can receive targeted offers, improving conversion rates and reducing spam.

Raw transactions are converted into **RFM features** (Recency, Frequency, Monetary) per customer. Three unsupervised clustering algorithms are then applied and compared using internal metrics and business interpretation.

## Dataset
- **Name:** Online Retail II (UCI Machine Learning Repository)
- **Link:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Kaggle mirror:** https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
- **Size:** about 1 million transactions (2009 to 2011), 8 columns (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country)
- **Used:** UK customers only. After cleaning, **5,350 unique customers** remain.

> The dataset file is not included in this repository because of its size. Download it from the link above.

## Methodology

### 1. Data Cleaning and EDA
- Kept only `Country == 'United Kingdom'`
- Dropped rows with null `CustomerID`
- Removed returns and cancellations (`Quantity <= 0`) and invalid prices (`UnitPrice <= 0`)
- Created `TotalPrice = Quantity x UnitPrice`
- Analysed distributions (log scale), monthly transaction trend (pre-Christmas seasonal peak) and the 80/20 revenue concentration

### 2. RFM Feature Engineering
| Feature | Definition |
|---|---|
| Recency | Days between the customer's last purchase and the reference date 2011-12-31 |
| Frequency | Number of unique invoices |
| Monetary | Total spend (sum of TotalPrice) |

### 3. Preprocessing
1. **Outlier capping:** winsorised each feature at Q3 + 3 x IQR (no rows dropped)
2. **Log transform:** `log1p` applied to Frequency and Monetary to reduce right skew
3. **Scaling:** `StandardScaler` on Recency, Frequency_log and Monetary_log (all three algorithms are distance-based)

### 4. Clustering
| Algorithm | Tuning approach | Final setting |
|---|---|---|
| K-Means | Elbow curve and Silhouette score for k = 2 to 10 | k = 4 (elbow point; k = 2 had the highest silhouette but was too coarse for marketing) |
| Agglomerative | Ward dendrogram, cut at distance of about 39.04; compared ward, complete and average linkage | Ward, 4 clusters (best silhouette 0.371; complete and average gave clusters of 41 and 7 customers) |
| DBSCAN | k-NN distance plot (k = 5), then grid over eps and min_samples | eps = 0.3, min_samples = 10 (eps = 0.5 had the best silhouette but found only 2 clusters) |

### 5. Evaluation
Silhouette Score, Davies-Bouldin Index and Calinski-Harabasz Index, plus a K-Means stability check over 5 random seeds.

## Results

| Algorithm | Hyperparameters | Clusters | Silhouette (higher is better) | Davies-Bouldin (lower is better) | Calinski-Harabasz (higher is better) | Noise % |
|---|---|---|---|---|---|---|
| **K-Means** | k=4, n_init=20 | 4 | **0.3735** | 0.9054 | **5797.0** | 0% |
| Agglomerative | ward, n_clusters=4 | 4 | 0.3708 | **0.8786** | 4867.1 | 0% |
| DBSCAN | eps=0.3, min_samples=10 | 3 | 0.2011 | 1.7719 | 2809.4 | 4.41% |

- **Ranking:** 1) K-Means, 2) Agglomerative (Ward), 3) DBSCAN
- **K-Means stability:** Silhouette was 0.3735 for all 5 seeds (0, 7, 21, 42, 99), mean 0.3735 with standard deviation 0.0, so the result is fully stable.
- DBSCAN metrics are computed on non-noise points only, so they are slightly flattered.

## Customer Personas

### K-Means (recommended model)
| Persona | Avg Recency (days) | Avg Frequency | Avg Monetary | Share of customers | Description |
|---|---|---|---|---|---|
| Champions | 65 | 15.1 | 5,233 | 20.7% | Recent, frequent, highest spenders. VIP customers. |
| Loyal Customers | 140 | 4.8 | 1,551 | 31.5% | Repeat buyers with solid spend. Can be moved up to Champions. |
| New Customers | 131 | 1.6 | 363 | 21.3% | Bought once or twice recently, low spend, still unproven. |
| Hibernating | 521 | 1.6 | 455 | 26.5% | Inactive for over a year, low spend. Essentially lapsed. |

### Marketing action per segment
| Segment | Action |
|---|---|
| Champions | Invite to an exclusive loyalty programme |
| Loyal Customers | Personalised offers to move them up to Champions |
| New Customers | Welcome offer for a second purchase |
| Hibernating | Win-back coupon with a 7-day expiry |

### Other algorithms
- **Agglomerative (Ward):** Champions, Promising Customers, **At-Risk Customers** (formerly good customers inactive for about 455 days) and Hibernating. It found an At-Risk group that K-Means did not separate.
- **DBSCAN:** Loyal/Active High-Value, At-Risk, Hibernating, plus 236 noise customers (4.4%) who are mostly lapsed big spenders and should get personal win-back outreach.

## Screenshots

| Elbow and Silhouette (K-Means) | Dendrogram (Ward) |

<img width="1263" height="487" alt="image" src="https://github.com/user-attachments/assets/3c6ff371-ca3e-4469-a7b1-12b20c8fb32b" />
<img width="1257" height="593" alt="image" src="https://github.com/user-attachments/assets/5ee8dc2c-2358-4cd0-8037-8e1e1255dcc9" />


| k-NN distance plot (DBSCAN) | K-Means 3D clusters |

<img width="857" height="473" alt="image" src="https://github.com/user-attachments/assets/939035f9-054d-4348-a273-6c5bbf17f949" />
<img width="727" height="717" alt="image" src="https://github.com/user-attachments/assets/549c0d72-e871-48bd-aad8-cdaade5a68a4" />


## Project Structure
```
customer-segmentation-unsupervised-learning/
├── CustomerSegmentation_UnsupervisedLearning.ipynb   # fully executed notebook
├── rfm_scaler.pkl                                    # saved StandardScaler
├── customer_segmentation_model.pkl                   # saved K-Means model
├── summary_report.md                                 # short written report
├── requirements.txt                                  # dependencies
├── images/                                           # screenshots used in this README
└── README.md
```

## How to Run
1. Clone the repository
```bash
   git clone https://github.com/ranu77/customer-segmentation-unsupervised-learning.git
   cd customer-segmentation-unsupervised-learning
```
2. Install the dependencies
```bash
   pip install -r requirements.txt
```
3. Download `online_retail_II.xlsx` from the dataset link above and place it in the project folder.
4. Open `CustomerSegmentation_UnsupervisedLearning.ipynb` and run all cells from top to bottom. Loading the Excel file takes 1 to 2 minutes.

## Using the Saved Model
```python
import joblib
import numpy as np
import pandas as pd

scaler = joblib.load('rfm_scaler.pkl')
model = joblib.load('customer_segmentation_model.pkl')
persona_map = {0: 'Champions', 3: 'Loyal Customers', 1: 'New Customers', 2: 'Hibernating'}

def predict_segment(recency, frequency, monetary):
    x = pd.DataFrame([[recency, np.log1p(frequency), np.log1p(monetary)]],
                     columns=['Recency', 'Frequency_log', 'Monetary_log'])
    label = int(model.predict(scaler.transform(x))[0])
    return label, persona_map[label]

print(predict_segment(10, 20, 8000))   # (0, 'Champions')
```

## Key Findings
- K-Means gave the most compact, well-separated and balanced segments, and was identical across all random seeds.
- Hierarchical clustering agreed with K-Means on the top segment (1,108 of 1,109 Champions matched) and additionally surfaced an At-Risk group.
- DBSCAN performed worst on internal metrics, but its noise points highlight unusual customers worth reviewing individually.
- Outlier capping pulled extreme VIPs back into the main clusters, so DBSCAN's noise is mostly lapsed high-value customers rather than ultra-high spenders.

## Author
**Devanshi Kanthariya**
