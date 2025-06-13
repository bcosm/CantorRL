import pickle
import sys

with open('./quantconnect/model_files/normalization_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

print('Top-level type:', type(stats))
print('Top-level keys:', list(stats.keys()) if hasattr(stats, 'keys') else None)

for k, v in (stats.items() if hasattr(stats, 'items') else []):
    print(f'Key: {k}, type: {type(v)}')
    if hasattr(v, 'keys'):
        print(f'  Subkeys: {list(v.keys())}')
    elif isinstance(v, (list, tuple)):
        print(f'  Length: {len(v)}')
    else:
        print(f'  Value: {v}')
