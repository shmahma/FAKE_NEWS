import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(fake_path="data/Fake.csv", true_path="data/True.csv"):
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)

    fake["label"] = 0
    true["label"] = 1

    df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    df["content"] = df["title"] + " " + df["text"]

    return train_test_split(df["content"], df["label"], test_size=0.2, random_state=42)
