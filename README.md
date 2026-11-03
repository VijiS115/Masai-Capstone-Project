# Masai-Capstone-Project
# Project README & Operational Blueprint

## 1. How to Run
To execute the pipeline and score new shipments from a fresh clone, run the following commands in order:

```bash
# 1. Install pinned scikit-learn, pandas, numpy, and joblib
pip install -r requirements.txt

# 2. Package and serialize the final champion classifier
python package_model.py

# 3. Generate predictions and map operational actions on new shipments
python predict_delay.py
```

> **Version Compatibility Warning:** The serialized model artifact `models/delay_model.pkl` is tightly bound to `scikit-learn==1.6.1`. If the installed scikit-learn version on the environment differs, the automated grading system will automatically run `python package_model.py` to regenerate the model instead of loading the committed pickle.

---

## 2. Operations Recommendation

### A. High-Risk 'Escalate' Segment Hubs
The following **9 hubs** fall into the **Escalate** segment with a staggering average delay rate of **46.00%** (mean delivery time: 64.65 hours). These require immediate operational focus:
- **HUB13 (Nagpur)**, **HUB15 (Kochi)**, **HUB17 (Patna)**, **HUB18 (Ranchi)**, **HUB19 (Guwahati)**, **HUB20 (Raipur)**, **HUB21 (Bhubaneswar)**, **HUB23 (Siliguri)**, **HUB24 (Agartala)**.

### B. Tier Label vs. Segment Mismatch
Our K-Means clustering analysis isolated **4 hubs** whose actual operational behavior deviates from their administrative tier designation:
1. **HUB07 (Kolkata)**: Designated as **Tier 1**, but acts like **Tier 2 (Watch)** with an elevated delay rate of **14.00%**.
2. **HUB13 (Nagpur)** and **HUB15 (Kochi)**: Designated as **Tier 2**, but act like **Tier 3 (Escalate)** with severe delay rates of **42.00%** and **40.00%** respectively.
3. **HUB22 (Dehradun)**: Designated as **Tier 3**, but acts like **Tier 2 (Watch)** with a lower delay rate of **20.00%**.

**Operational Impact:** Administrative "tier labels" must **not** be used to define SLAs or resources. Instead, rely strictly on the dynamic **K-Means Segment labels** (`outputs/hub_segments.csv`) to deploy physical recovery playbooks.

### C. Alert Threshold & Classifier Power
Using our chosen alert threshold of **0.5**, our classifier achieves the following performance profiles on unseen test data:
- **Precision (68.89%):** Out of every 10 flagged parcels, approximately **7 are truly delayed**. This means 3 are false alarms, representing a highly manageable operational overhead.
- **Recall (75.61%):** Out of every 10 truly delayed parcels, **7 to 8 are caught** before they breach their SLA, giving local teams ample warning to intervene.

### D. Real-Time Action Metrics on 48 New Shipments
Out of the 48 incoming test shipments scored by `predict_delay.py`:
- **15 parcels** were flagged as high-risk.
- **5 flagged parcels** belong to the *Escalate* segment and were routed to **"Escalate to hub manager"**.
- **10 flagged parcels** belong to *Watch* or *Standard* segments and were routed to **"Proactive customer call"**.
- **33 parcels** require **"No action"**.

### E. Why We Maintained K = 3
Our separation diagnostic (`part_d_separation.csv`) shows that K=3 is the mathematical sweet spot. Increasing K from 3 to 4 or 5 offers minimal marginal improvement in cluster separation (`across_over_within` rises from 5.39 to 6.30, while splitting the small 24-hub sample into tiny clusters of size 2 or 3). K=3 ensures high operational stability and maps cleanly to the three physical playbooks desired by regional managers.

---

## 3. Production Deployment Plan

### A. Production-Readiness Mapping
1. **Consistency:** Guaranteed by `models/impute_values.csv` and `models/feature_order.csv`, forcing identical preprocessing and sequence during both training and live inference.
2. **Reproducibility:** Ensured by pinning scikit-learn version and hardcoding the random seed (`random_state=42`) in all training/split stages.
3. **Reliability:** Built-in scanner-fault filtering handles corrupted weights (>100 kg) gracefully, substituting them with historical medians.
4. **Performance:** Pipeline leverages lightweight scikit-learn structures, executing inference in less than 50 milliseconds on the 48-shipment batch.
5. **Maintainability:** The modular design isolates shipment preprocessing in `clean_shipments` and feature synthesis in `build_features`.

### B. Monitoring Framework
We recommend monitoring the following three operational metrics on a weekly rolling basis:
1. **Weekly Delay Share:** Trigger warning if the delayed share of delivered parcels exceeds **30.0%** (indicates localized network-wide bottlenecks).
2. **Weekly Flagged Rate:** Trigger warning if flagged parcels exceed **35.0%** of total volume (indicates potential scanner drift or systemic routing changes).
3. **Realized Recall:** Trigger warning if realized recall on scored parcels with known delivery times drops below **70.0%** (indicates classification performance decay).

### C. Retraining & Seasonality Management
- **Schedule:** Retrain the Random Forest classifier **monthly** or when a **5% absolute performance degradation** is observed in realized recall.
- **Handling Sale Events:** The 15-28 September sale window (`sale_event_flag`) must **not** be compared against normal weeks. Treat peak sales as distinct seasons; evaluate model performance on holiday/sale periods separately to prevent skewing baseline model expectations.
- **Data Drift Identification:** Monitor shifts in underlying feature distributions (e.g., if the average transit distance increases by >15% due to new route alignments). This is identified using the Population Stability Index (PSI); triggering a retraining cycle when PSI > 0.2.

---

## 4. System Limitations
1. **Synthetic Data Boundaries:** The environment relies on clean, synthetically simulated distributions. In real-world networks, unrecorded events (e.g., driver shortages, vehicle breakdown) will degrade regression and classification accuracy.
2. **Small Validation/Test Slices:** With only 41 delayed shipments in our test set, performance metrics (like recall) are highly sensitive to small variations. Missing just 2 more delays causes recall to slide significantly.
3. **Single Validation Split Tuning:** The alert threshold (0.5) was tuned against a single 180-row validation partition. Over time, seasonal variance may require a dynamic threshold tuning strategy.
