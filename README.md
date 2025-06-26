# credit-risk-model
**Influence of Basel II on Model Interpretability**  

The Basel II Capital Accord emphasizes risk-sensitive capital allocation, requiring financial institutions to measure credit risk more accurately and justify the internal models used for regulatory capital estimation. This regulatory framework mandates transparency, auditability, and stress testing capabilities.

As a result, models used for credit scoring must be:

- Interpretable: Decision-makers and regulators need to understand how inputs drive outputs.

- Well-documented: Model development, assumptions, limitations, and validation procedures must be traceable.

**Need for Proxy Default Variables and Associated Risks**  

In the absence of a clear "default" indicator analysts must construct proxy variables, such as:

* 90+ days past due (DPD)

* Charge-offs

* Collection referrals

This is essential for supervised learning, where labeled outcomes are required to train and validate models.

However, business risks include:

* Label leakage: If the proxy doesn't accurately represent true default, the model may misclassify risk.

* Bias and fairness issues: Proxies may systematically disadvantage certain customer segments (e.g., informal income earners in developing economies).

 * non-compliance: Misaligned definitions with regulatory standards can result in penalties or invalidation of risk-weighted assets (RWAs).

In regulated financial environments, particularly under the Basel II Accord and supervisory expectations, a careful balance must be struck between model complexity and regulatory compliance. Interpretable models, such as logistic regression with Weight of Evidence (WoE) encoding, are widely used due to their high transparency, allowing clear understanding of how input variables affect the credit score. They are also easier to justify to regulators, have lower operational costs, and are simpler to audit for fairness and bias. However, their performance may be moderate, as they can underfit complex or non-linear data patterns. In contrast, complex models like Gradient Boosting or XGBoost offer higher predictive performance, capturing intricate interactions in the data. Yet, they often function as black boxes, requiring model explainers such as SHAP to achieve transparency. These models also come with higher operational costs and greater difficulty in demonstrating fairness, which poses challenges in regulatory contexts. To address these trade-offs, many institutions are adopting hybrid modeling strategies—utilizing complex models for initial screening or feature engineering, while relying on interpretable models for final decision-making. This approach enables organizations to benefit from enhanced performance without compromising regulatory transparency and accountability.







