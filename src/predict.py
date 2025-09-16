from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

model = DistilBertForSequenceClassification.from_pretrained("models/fake_news_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("models/fake_news_model")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "Fake" if pred == 0 else "True"

if __name__ == "__main__":
    example_true = "As U.S. budget fight looms, Republicans flip their fiscal script: WASHINGTON (Reuters) – The leader of a conservative Republican group in the U.S. Congress, who recently supported a major national debt increase to fund tax cuts, described himself as a 'fiscal conservative' on Sunday and called for budget restraint in 2018."
    example_fake = "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing: Donald Trump just couldn t wish all Americans a Happy New Year and leave it at that"
    print(predict(example_true))
    print(predict(example_fake))
