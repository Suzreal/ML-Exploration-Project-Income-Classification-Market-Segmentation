import numpy as np
import pandas as pd
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


df = pd.read_csv('cleaned_data.csv')
df['age_bucket'] = pd.cut(df['age'],bins=[-np.inf, 24, 54, np.inf], labels=['young_<=24', 'working_25_54', 'older_55+'])
edu = df['education'].astype(str).str.strip().str.lower()

def edu_bucket_fn(s):
    if 'children' in s:
        return 'child'
    if 'bachelors' in s or 'masters' in s or 'prof school' in s or 'doctorate' in s:
        return 'bachelor_plus'
    if 'high school graduate' in s or 'some college' in s or 'associates' in s:
        return 'hs_some_college'
    return 'lt_hs'

df['edu_bucket'] = edu.apply(edu_bucket_fn)
ws = df['full_or_part_time_employment_stat'].astype(str).str.strip().str.lower()

df['work_bucket'] = np.select(
    [
        ws.str.contains('children') | ws.str.contains('armed forces'),
        ws.str.contains('full-time schedules'),
        ws.str.contains('pt '),
        ws.str.contains('unemployed'),
        ws.str.contains('not in labor force'),
    ],
    [
        'child_or_armed_forces',
        'full_time',
        'part_time',
        'unemployed',
        'not_in_labor_force',
    ],
    default='other'
)

drop_cols = [c for c in df.columns if c in ['label', 'weight', 'year'] or c.startswith('segment_')]
X_seg = df.drop(columns=drop_cols, errors='ignore')
w_seg = df['weight'].astype(float)


numerical_col = X_seg.select_dtypes(include=[np.number]).columns.tolist()
categorical_col = [c for c in X_seg.columns if c not in numerical_col]

numerical_tran = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(with_mean=False))
])

cate_tran = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=50, sparse_output=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numerical_tran, numerical_col),
        ('cat', cate_tran, categorical_col),
    ],
    remainder='drop'
)


svd = TruncatedSVD(n_components=50, random_state=42)


X_enc = preprocess.fit_transform(X_seg)
X_red = svd.fit_transform(X_enc)

K = 5
km = KMeans(n_clusters=K, random_state=42, n_init='auto')
cluster_id = km.fit_predict(X_red)

df['cluster_ml'] = cluster_id

print(df.columns)

df.to_csv('final_segmentation_data.csv')