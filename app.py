from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
import pandas as pd
import joblib

# =====================================================
# Load trained Tier-1 model
# =====================================================
model = joblib.load("attrition_model2.pkl")

app = FastAPI(
    title="Employee Attrition Risk API",
    description="Returns attrition probability using Tier-1 core HRMS features.",
    version="1.1.0"
)

# =====================================================
# ENUMS
# =====================================================

class DepartmentEnum(str, Enum):
    HUMAN_RESOURCES = "Human Resources"
    RESEARCH_AND_DEVELOPMENT = "Research & Development"
    SALES = "Sales"


class JobRoleEnum(str, Enum):
    HEALTHCARE_REPRESENTATIVE = "Healthcare Representative"
    HUMAN_RESOURCES = "Human Resources"
    LABORATORY_TECHNICIAN = "Laboratory Technician"
    MANAGER = "Manager"
    MANUFACTURING_DIRECTOR = "Manufacturing Director"
    RESEARCH_DIRECTOR = "Research Director"
    RESEARCH_SCIENTIST = "Research Scientist"
    SALES_EXECUTIVE = "Sales Executive"
    SALES_REPRESENTATIVE = "Sales Representative"


class GenderEnum(str, Enum):
    FEMALE = "Female"
    MALE = "Male"


class MaritalStatusEnum(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"


class OverTimeEnum(str, Enum):
    YES = "Yes"
    NO = "No"

# =====================================================
# INPUT SCHEMA
# =====================================================

class EmployeeInput(BaseModel):
    Age: int = Field(..., example=32)
    DistanceFromHome: int = Field(..., example=10)
    JobLevel: int = Field(..., example=2)
    MonthlyIncome: float = Field(..., example=6500)
    NumCompaniesWorked: int = Field(..., example=3)
    TotalWorkingYears: int = Field(..., example=8)
    YearsAtCompany: int = Field(..., example=5)
    YearsInCurrentRole: int = Field(..., example=3)
    YearsSinceLastPromotion: int = Field(..., example=2)
    YearsWithCurrManager: int = Field(..., example=2)

    Department: DepartmentEnum
    JobRole: JobRoleEnum
    Gender: GenderEnum
    MaritalStatus: MaritalStatusEnum
    OverTime: OverTimeEnum

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================================================
# METADATA
# =====================================================

@app.get("/metadata")
def metadata():
    return {
        "model_type": "Logistic Regression",
        "output": "Probability of attrition (0 to 1)",
        "categorical_allowed_values": {
            "Department": [e.value for e in DepartmentEnum],
            "JobRole": [e.value for e in JobRoleEnum],
            "Gender": [e.value for e in GenderEnum],
            "MaritalStatus": [e.value for e in MaritalStatusEnum],
            "OverTime": [e.value for e in OverTimeEnum]
        }
    }

# =====================================================
# PREDICTION ENDPOINT
# =====================================================

@app.post("/predict")
def predict(employee: EmployeeInput):

    df = pd.DataFrame([employee.dict()])

    probability = model.predict_proba(df)[0][1]

    return {
        "attrition_probability": round(probability, 4)
    }
