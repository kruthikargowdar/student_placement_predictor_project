import pickle

with open("model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("Metadata type:", type(metadata))
print("Metadata content:", metadata)
