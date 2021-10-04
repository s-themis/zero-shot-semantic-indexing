import json
import pandas as pd

from collections import Counter

with open('data/test_2006.json') as f:
    data = json.load(f)

df = pd.DataFrame.from_records(data['documents'])
print(f"Number of documents: {len(df)}")

name_counts = Counter()
for names in df['Descriptor_names']:
    name_counts.update(names)
print(name_counts.most_common(10))

ui_counts = Counter()
for uis in df['Descriptor_UIs']:
    ui_counts.update(uis)
print(ui_counts.most_common(10))
