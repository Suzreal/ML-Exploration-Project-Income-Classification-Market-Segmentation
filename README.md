# ML-Exploration-Project-Income-Classification-Market-Segmentation

# User guide

- Please use *requirement.txt* to creat the environment if needed
- Please make sure cleaned_data.csv is under the same path of the 2 py script
    - To run in terminal: `python segementation.py` and  `classification.py`

The guide of the following files:


```bash
.
├── classification.py -> the final draft of income classification model
├── requirement.txt -> environment requirement
├── report.pdf -> report
├── segementation.py -> the final draft of segmentation model
├── Part1_classifier.ipynb -> classification model developing notebook
├── Part2_segmentation.ipynb -> segmentation model developing notebook
├── cluster_diff -> detailed ML generated cluster differences
├── cleaned_data.csv -> cleaned_data
├── Data Cleaning/
│   ├── census-bureau.columns -> raw column
│   └── census-bureau.data -> raw data
│   ├── Data_cleaning.ipynb -> data cleaning notebook
```

# Summary

**Part 1 Pipeline:**

![alt text](image.png)

**Part 1 Result(validation set):**
![alt text](image-2.png)

**Part 1 Result(final model on test set):**

- Decesion Threshold: 0.3
- Accuracy: 0.9485
- Precision: 0.6168
- Recall: 0.6268
- F1: 0.6218
- Roc_auc: 0.9509


**Part 2 Pipeline(ML model):**

![alt text](image-1.png)

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

![alt text](image-3.png)

