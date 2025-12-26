# ML-Exploration-Project-Income-Classification-Market-Segmentation

# User guide

- Please use *requirement.txt* to create the environment if needed
- Please make sure cleaned_data.csv is under the same path as the 2 py scripts
    - To run in terminal: `python segmentation.py` and  `classification.py`.

The guide of the following files:


```bash
.
├── classification.py -> the final draft of the income classification model
├── requirement.txt -> environment requirement
├── report.pdf -> report
├── segmentation.py -> the final draft of the segmentation model
├── Part1_classifier.ipynb -> classification model developing notebook
├── Part2_segmentation.ipynb -> segmentation model developing notebook
├── cluster_diff -> detailed ML-generated cluster differences
├── cleaned_data.csv -> cleaned_data
├── Data Cleaning/
│   ├── census-bureau.columns -> raw column
│   └── census-bureau.data -> raw data
│   ├── Data_cleaning.ipynb -> data cleaning notebook
```

# Summary

**Part 1 Pipeline:**

<img width="792" height="219" alt="image" src="https://github.com/user-attachments/assets/8f3c7218-6ab1-463f-acd9-5aeeebad13b8" />


**Part 1 Result(validation set):**
<img width="792" height="310" alt="image-2" src="https://github.com/user-attachments/assets/da01aadb-9d2d-4583-bb1a-8e63d2174001" />


**Part 1 Result(final model on test set):**

- Decision Threshold: 0.3
- Accuracy: 0.9485
- Precision: 0.6168
- Recall: 0.6268
- F1: 0.6218
- Roc_auc: 0.9509


**Part 2 Pipeline(ML model):**

<img width="788" height="223" alt="image-1" src="https://github.com/user-attachments/assets/130c1b25-058b-4fff-bdc7-cdc0af3e8e80" />


Final cluster summary and business meaning:
- Cluster 0: 7.31% pop, mean age 21.0, ≥$50K rate 0.09%
    - Young Adults / Students & Early Career (low income)
- Cluster 1: 25.54% pop, mean age 38.4, ≥$50K rate 12.44%
    - Prime-Age Workforce (Segment A)
- Cluster 2: 25.34% pop, mean age 38.2, ≥$50K rate 11.14%
    - Prime-Age Workforce (Segment B)
- Cluster 3: 22.77% pop, mean age 6.9, ≥$50K rate 0.00%
    - Children / Dependents
- Cluster 4: 19.05% pop, mean age 62.4, ≥$50K rate 2.09%
    - Older / Retired or Not Working

For more details of differences in clusters, please check the `cluster_diff.csv`

Heatmap(rule-based vs ML segments)

<img width="936" height="490" alt="image-3" src="https://github.com/user-attachments/assets/93582073-46a7-4989-b0c6-0b7c6959f9ba" />


