import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset


# --- Step 1: Synthetic Dataset Generator ---
symptoms = [
    "headache", "fever", "sore throat", "stomach pain", "cough", "dizziness",
    "back pain", "tiredness", "cold", "chest pain", "allergy", "vomiting",
    "diarrhea", "ear pain", "toothache", "knee pain", "anxiety", "stress",
    "skin rash", "muscle pain", "shortness of breath", "eye pain"
]


advice = [
    "Please take rest and drink plenty of fluids.",
    "Consider taking paracetamol as prescribed.",
    "Try warm salt water gargles.",
    "Eat a light diet and avoid oily food.",
    "Stay hydrated and monitor your symptoms.",
    "If symptoms persist, consult a doctor.",
    "Practice deep breathing and relaxation.",
    "Use an ice pack for relief.",
    "Take proper sleep and balanced diet.",
    "Avoid dust and keep your room ventilated.",
    "Do gentle stretching exercises."
]


def format_conversation(patient, doctor):
    return f"### Patient:\nI have {patient}.\n\n### Doctor:\n{doctor}\n"


# generate ~80 samples
data = []
for _ in range(80):
    symptom = random.choice(symptoms)
    doctor_advice = random.choice(advice)
    data.append({"text": format_conversation(symptom, doctor_advice)})


dataset = Dataset.from_list(data)


# --- Step 2: Dataset Inspection ---
print("Dataset size:", len(dataset))
print("\nSample 3 random conversations:\n")
for sample in random.sample(data, 3):
    print(sample["text"])
    print("-" * 60)


# --- Step 3: Split Train/Test (90/10) ---
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]


print(f"\nTrain size: {len(train_dataset)} | Test size: {len(test_dataset)}")


# --- Step 4: Load tokenizer & model ---
model_name = "distilgpt2"   # small for demo; replace with deepseek if GPU available
tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(model_name)


# --- Step 5: Tokenization ---
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)


# --- Step 6: Training setup ---
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",    # changed from evaluation_strategy
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    logging_steps=10,
    save_total_limit=1,
    learning_rate=5e-5,
    warmup_steps=5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)


# --- Step 7: Train + Evaluate ---
trainer.train()
eval_results = trainer.evaluate()
print("\nEvaluation Results:", eval_results)


# --- Step 8: Test Generation ---
input_text = "### Patient:\nI have headache.\n\n### Doctor:\n"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)


outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8
)


print("\nGenerated Response:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
