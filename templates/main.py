import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import shutil
import os
import re
import pdfplumber
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
import plotly.express as px

# Setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")
os.makedirs("temp", exist_ok=True)

# Fields to extract
FIELDS = [
    'WBC', 'RBC', 'Hemoglobin', 'Hematocrit', 'Platelets',
    'Glucose', 'cholesterol', 'Triglycerides', 'LDL', 'HDL', 'calcium',
    'TSH', 'FT4', 'Ferritin', 'Vitamin D', 'B12',
]

# Utility functions for extraction and scaling 
def extract_numeric_value(text):
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)
    if match:
        return float(match.group(0))
    return None

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
    pattern = re.compile(rf"{field}\s*(\([^)]+\))?\s*[:\s]*([-+]?\d*\.\d+|\d+)", re.IGNORECASE)
    match = pattern.search(line)
    if match:
        return float(match.group(2))
    return None

def extract_from_text(text, results):
    for line in text.split("\n"):
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
    if results.get("WBC") and results["WBC"] > 1000:
        results["WBC"] /= 1000
    if results.get("RBC") and results["RBC"] > 1000:
        results["RBC"] /= 100
    if results.get("Platelets") and results["Platelets"] > 1000:
        results["Platelets"] /= 1000
    if results.get("WBC") and results["WBC"] > 20:
        results["WBC"] /= 10

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

# Load models
model_paths = {
    "🩸 Hematology": ("templates/model_hematology.keras", "templates/mlb_hematology.pkl", "templates/scaler_hematology.pkl", "templates/X1.pkl"),
    "🧪 Chemistry": ("templates/model_chemistry.keras", "templates/mlb_chemistry.pkl", "templates/scaler_chemistry.pkl", "templates/X2.pkl"),
    "🧬 Serology + Immunology": ("templates/model_immunology.keras", "templates/mlb_immunology.pkl", "templates/scaler_immunology.pkl", "templates/X3.pkl"),
}
models = {}
for name, (model_path, mlb_path, scaler_path, x_path) in model_paths.items():
    models[name] = (
        load_model(model_path),
        pickle.load(open(mlb_path, 'rb')),
        pickle.load(open(scaler_path, 'rb')),
        pickle.load(open(x_path, 'rb'))
    )

# Chart creation functions (using plotly)
def create_monthly_chart():
    df = pd.read_csv("templates/table_1.csv", index_col=0).dropna(how='all')
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[(df.index != '')]
    fig = px.bar(df, x=df.index, y='total', title="Total Cases by Disease (Monthly Data)")
    fig.update_layout(xaxis_title="Disease", yaxis_title="Total Cases", xaxis_tickangle=45)
    return fig.to_html(full_html=False)

def create_age_chart():
    df = pd.read_csv("templates/table_2.csv", index_col=0).dropna(how='all')
    df.columns = [c.strip() for c in df.columns]
    age_cols = ['0-4 years', '5-9 years', '10-19 years', '20-39 years', '40-59 years', '60+ years']
    df_age_sum = df[age_cols].sum()
    fig = px.pie(values=df_age_sum.values, names=df_age_sum.index, title="Cases by Age Groups")
    return fig.to_html(full_html=False)

def create_region_chart():
    df = pd.read_csv("templates/table_3.csv", index_col=0).dropna(how='all')
    df.columns = [c.strip() for c in df.columns]
    region_cols = ['north', 'beqaa', 'nabatieh', 'south', 'mount-lebanon', 'beirut', 'unknown']
    df_region_sum = df[region_cols].sum()
    fig = px.bar(x=df_region_sum.index, y=df_region_sum.values, title="Cases by Region")
    fig.update_layout(xaxis_title="Region", yaxis_title="Total Cases")
    return fig.to_html(full_html=False)

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    monthly_chart = create_monthly_chart()
    age_chart = create_age_chart()
    region_chart = create_region_chart()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "monthly_chart": monthly_chart,
        "age_chart": age_chart,
        "region_chart": region_chart
    })

@app.post("/predict")
async def predict_lab_data(file: UploadFile = File(...)):
    try:
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        input_data = extract_lab_data_from_pdf(file_path)

        predictions = {}
        for name, (model, mlb, scaler, X_ref) in models.items():
            df = pd.DataFrame([input_data])
            for col in X_ref.columns:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[X_ref.columns]
            df.fillna(X_ref.median(numeric_only=True), inplace=True)
            df_scaled = scaler.transform(df)
            pred_probs = model.predict(df_scaled)[0]
            top_indices = np.argsort(pred_probs)[-5:][::-1]
            top_predictions = {mlb.classes_[i]: float(pred_probs[i]) for i in top_indices}
            predictions[name] = top_predictions

        os.remove(file_path)
        return JSONResponse(content={"input_data": input_data, "predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
