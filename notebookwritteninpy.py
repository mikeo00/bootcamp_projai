from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import pdfplumber
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#pdf extraction
from datetime import datetime

FIELDS = [
    'WBC', 'RBC', 'Hemoglobin', 'Hematocrit', 'Platelets',
    'Glucose', 'cholesterol', 'Triglycerides', 'LDL', 'HDL','calcium',
    'TSH','FT4', 'Ferritin', 'Vitamin D', 'B12',
]

def extract_numeric_value(text):
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group(0))
    return None

from datetime import datetime

def extract_age_gender(text):
    age = None
    gender = None

    age_match = re.search(r'\bAge[:\s]*([0-9]{1,3})\b', text, re.IGNORECASE)
    if age_match:
        age = int(age_match.group(1))

    gender_match = re.search(r'\b(Male|Female)\b', text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).capitalize()

    text = text.replace('–', '-').replace('—', '-').replace('‒', '-')
    text = re.sub(r'\s+', ' ', text) 

    combo_match = re.search(
        r'\bDoB\s*/\s*Gender[:\s]*([0-9]{1,4}[-/.][0-9]{1,2}[-/.][0-9]{2,4})\s*/\s*(Male|Female)\b',
        text,
        re.IGNORECASE
    )
    if combo_match:
        dob_str = combo_match.group(1)
        gender = combo_match.group(2).capitalize()

        date_formats = [
            "%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d",
            "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d.%m.%Y", "%m.%d.%Y", "%Y.%m.%d",
            "%d-%m-%y", "%m-%d-%y", "%y-%m-%d",
            "%d/%m/%y", "%m/%d/%y", "%y/%m/%d",
            "%d.%m.%y", "%m.%d.%y", "%y.%m.%d",
        ]

        for fmt in date_formats:
            try:
                dob = datetime.strptime(dob_str, fmt)
                today = datetime.today()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                break
            except ValueError:
                continue

    if age is None:
        dob_match = re.search(
            r'\b(DOB|Date of Birth)[:\s]*([0-9]{1,4}[-/.][0-9]{1,2}[-/.][0-9]{2,4})\b',
            text,
            re.IGNORECASE
        )
        if dob_match:
            dob_str = dob_match.group(2)
            for fmt in date_formats:
                try:
                    dob = datetime.strptime(dob_str, fmt)
                    today = datetime.today()
                    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    break
                except ValueError:
                    continue

    return age, gender


def extract_value_from_line(line, field):
    pattern = re.compile(
        rf"{field}\s*(\([^)]+\))?\s*:\s*([-+]?\d*\.\d+|\d+)", re.IGNORECASE
    )
    match = pattern.search(line)
    if match:
        value_str = match.group(2)
        try:
            return float(value_str)
        except:
            return None
    return None

def extract_from_text(text, results):
    lines = text.split("\n")
    for line in lines:
        for field in FIELDS:
            if field.lower() in line.lower():
                value = extract_value_from_line(line, field)
                if value is not None:
                    results[field] = value

def extract_from_tables(tables, results):
    for table in tables:
        for row in table:
            if not row or len(row) < 2:
                continue
            name = str(row[0]).strip()
            value_text = str(row[1]).strip()
            for field in FIELDS:
                if field.lower() in name.lower():
                    value = extract_numeric_value(value_text)
                    if value is not None:
                        results[field] = value

def scale_values(results):
    if results["WBC"] and results["WBC"] > 1000:
        results["WBC"] = results["WBC"] / 1000
    if results["RBC"] and results["RBC"] > 1000:
        results["RBC"] = results["RBC"] / 100
    if results["Platelets"] and results["Platelets"] > 1000:
        results["Platelets"] = results["Platelets"] / 1000
    if results["WBC"] and results["WBC"] > 20:
        results["WBC"] = results["WBC"] / 10

def extract_lab_data_from_pdf(pdf_path):
    results = {field: None for field in FIELDS}
    full_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
                extract_from_text(text, results)

            tables = page.extract_tables()
            if tables:
                extract_from_tables(tables, results)

    age, gender = extract_age_gender(full_text)
    results['age'] = age
    results['gender'] = gender

    scale_values(results)

    return results

if __name__ == "__main__":
    pdf_file = input("Enter path to PDF file: ").strip()
    try:
        lab_data = extract_lab_data_from_pdf(pdf_file)
        print("\nExtracted Lab Data:")
        for k, v in lab_data.items():
            print(f"{k}: {v}")
        
        with open("extracted_lab_data.json", "w") as f:
            json.dump(lab_data, f, indent=4)
        print("\n✅ Data saved to 'extracted_lab_data.json'")

    except Exception as e:
        print(f"Error: {e}")
#load and preprocess Dataset 1
df1 = pd.read_csv('6s00_patient.csv')
df1 = df1.drop(columns=["PatientID"])
df1["Gender"] = df1["Gender"].map({"Male": 1, "Female": 0})
df1["Diagnosis"] = df1["Diagnosis"].apply(
    lambda x: [d.strip() for d in x.split(",")] if isinstance(x, str) else ["Normal"]
)

mlb1 = MultiLabelBinarizer()
diagnosis_encoded1 = mlb1.fit_transform(df1["Diagnosis"])
diagnosis_df1 = pd.DataFrame(diagnosis_encoded1, columns=mlb1.classes_)

df1 = df1.drop(columns=["Diagnosis"])
df1 = pd.concat([df1, diagnosis_df1], axis=1)

X1 = df1.drop(columns=mlb1.classes_)
y1 = df1[mlb1.classes_]

scaler1 = StandardScaler()
X_scaled1 = scaler1.fit_transform(X1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_scaled1, y1.values, test_size=0.2, random_state=42
)
#load and preprocess Dataset 2
df2 = pd.read_csv('metabolic_600k_patients.csv')
df2 = df2.drop(columns=["PatientID"])
df2["Gender"] = df2["Gender"].map({"Male": 1, "Female": 0})
df2["Diagnosis"] = df2["Diagnosis"].apply(
    lambda x: [d.strip() for d in x.split(",")] if isinstance(x, str) else ["Normal"]
)

mlb2 = MultiLabelBinarizer()
diagnosis_encoded2 = mlb2.fit_transform(df2["Diagnosis"])
diagnosis_df2 = pd.DataFrame(diagnosis_encoded2, columns=mlb2.classes_)

df2 = df2.drop(columns=["Diagnosis"])
df2 = pd.concat([df2, diagnosis_df2], axis=1)

X2 = df2.drop(columns=mlb2.classes_)
y2 = df2[mlb2.classes_]

scaler2 = StandardScaler()
X_scaled2 = scaler2.fit_transform(X2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled2, y2.values, test_size=0.2, random_state=42
)
#load and preprocess Dataset 3
df3 = pd.read_csv('newwwthyr.csv')
df3 = df3.drop(columns=["PatientID"])
df3["Gender"] = df3["Gender"].map({"Male": 1, "Female": 0})
df3["Diagnosis"] = df3["Diagnosis"].apply(
    lambda x: [d.strip() for d in x.split(",")] if isinstance(x, str) else ["Normal"]
)

mlb3 = MultiLabelBinarizer()
diagnosis_encoded3 = mlb3.fit_transform(df3["Diagnosis"])
diagnosis_df3 = pd.DataFrame(diagnosis_encoded3, columns=mlb3.classes_)

df3 = df3.drop(columns=["Diagnosis"])
df3 = pd.concat([df3, diagnosis_df3], axis=1)

X3 = df3.drop(columns=mlb3.classes_)
y3 = df3[mlb3.classes_]

scaler3 = StandardScaler()
X_scaled3 = scaler3.fit_transform(X3)

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X_scaled3, y3.values, test_size=0.2, random_state=42
)
#model Setup for Dataset 1

filepath1 = "hematology-best-{epoch:02d}-{val_accuracy:.2f}.keras"

checkpoint1 = ModelCheckpoint(
    filepath1,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stop1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model1 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train1.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train1.shape[1], activation='sigmoid')
])

model1.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history1 = model1.fit(
    X_train1, y_train1,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop1],
    verbose=1
)
#model Setup for Dataset 2
filepath2 = "chemistry-best-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model2 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train2.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train2.shape[1], activation='sigmoid')
])

model2.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history2 = model2.fit(
    X_train2, y_train2,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop2],
    verbose=1
)
#model Setup for Dataset 3
filepath3 = "serology-best-{epoch:02d}-{val_accuracy:.2f}.keras"
checkpoint3 = ModelCheckpoint(filepath3, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop3 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model3 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train3.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train3.shape[1], activation='sigmoid')
])

model3.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history3 = model3.fit(
    X_train3, y_train3,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop3],
    verbose=1
)
import pickle

model1.save("model_hematology.keras")
model2.save("model_chemistry.keras")
model3.save("model_immunology.keras")

with open("scaler_hematology.pkl", "wb") as f:
    pickle.dump(scaler1, f)

with open("scaler_chemistry.pkl", "wb") as f:
    pickle.dump(scaler2, f)

with open("scaler_immunology.pkl", "wb") as f:
    pickle.dump(scaler3, f)

with open("mlb_hematology.pkl", "wb") as f:
    pickle.dump(mlb1, f)

with open("mlb_chemistry.pkl", "wb") as f:
    pickle.dump(mlb2, f)

with open("mlb_immunology.pkl", "wb") as f:
    pickle.dump(mlb3, f)
with open("X1.pkl", "wb") as f:
    pickle.dump(X1, f)
with open("X2.pkl", "wb") as f:
    pickle.dump(X2, f)
with open("X3.pkl", "wb") as f:
    pickle.dump(X3, f)
from tensorflow.keras.models import load_model

model1 = load_model("model_hematology.keras")
model2 = load_model("model_chemistry.keras")
model3 = load_model("model_immunology.keras")

with open("scaler_hematology.pkl", "rb") as f:
    scaler1 = pickle.load(f)

with open("scaler_chemistry.pkl", "rb") as f:
    scaler2 = pickle.load(f)

with open("scaler_immunology.pkl", "rb") as f:
    scaler3 = pickle.load(f)

with open("mlb_hematology.pkl", "rb") as f:
    mlb1 = pickle.load(f)

with open("mlb_chemistry.pkl", "rb") as f:
    mlb2 = pickle.load(f)

with open("mlb_immunology.pkl", "rb") as f:
    mlb3 = pickle.load(f)
with open("X1.pkl", "rb") as f:
    X1 = pickle.load(f)
with open("X2.pkl", "rb") as f:
    X2 = pickle.load(f)
with open("X3.pkl", "rb") as f:
    X3 = pickle.load(f)
import json
import numpy as np
import pandas as pd

with open("extracted_lab_data.json", "r") as f:
    input_data = json.load(f)

models = {
    "🩸 Hematology": (model1, mlb1, scaler1, X1),
    "🧪 Chemistry": (model2, mlb2, scaler2, X2),
    "🧬 Serology + Immunology": (model3, mlb3, scaler3, X3)
}

threshold = 0.5
top_n = 5

for name, (mdl, mlb, scaler, X_ref) in models.items():
    print(f"\n🔍 {name} Predictions:")

    input_df = pd.DataFrame([input_data])

    for col in X_ref.columns:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[X_ref.columns]
    input_df.fillna(X_ref.median(numeric_only=True), inplace=True)

    input_scaled = scaler.transform(input_df)

    prediction = mdl.predict(input_scaled)[0]
    top_indices = np.argsort(prediction)[-top_n:][::-1]

    print("Top Predictions (sorted by confidence):")
    for i in top_indices:
        print(f"{mlb.classes_[i]}: {prediction[i]:.4f}")

    predicted_labels = [mlb.classes_[i] for i, p in enumerate(prediction) if p >= threshold]

    if predicted_labels:
        print("Predicted Diagnosis:")
        for label in predicted_labels:
            print(f"- {label}")
    else:
        print("No diagnosis exceeded threshold.")
