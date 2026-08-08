"""
Dashboard Dự đoán & Cảnh báo Sớm Nghỉ Việc Nhân Sự — bản Gradio
-----------------------------------------------------------------
Chạy local:  python app/gradio_app.py   (từ thư mục gốc project)
Deploy free: đẩy file này + models/ + requirements.txt lên Hugging Face Spaces
             (chọn SDK = Gradio), Spaces sẽ tự chạy demo.launch() bên trong.
"""
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import shap
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Load model, scaler, schema — load 1 lần khi app khởi động
# ----------------------------------------------------------------------------
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
with open("models/threshold.txt") as f:
    THRESHOLD = float(f.read())
with open("models/model_name.txt") as f:
    MODEL_NAME = f.read()
with open("models/feature_schema.json", encoding="utf-8") as f:
    schema = json.load(f)

explainer = shap.TreeExplainer(model.named_steps["clf"])
MEDIUM_THRESHOLD = THRESHOLD / 3

# Các cột numeric không có ô nhập riêng trên form -> tự điền giá trị trung vị
FORM_NUMERIC_COLS = [
    "Age", "DistanceFromHome", "Education", "JobLevel", "MonthlyIncome",
    "StockOptionLevel", "PercentSalaryHike", "PerformanceRating", "JobSatisfaction",
    "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement",
    "WorkLifeBalance", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
    "TrainingTimesLastYear",
]
REMAINING_NUMERIC_COLS = [c for c in schema["numeric_cols"] if c not in FORM_NUMERIC_COLS]


def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Chuyển dữ liệu THÔ (định dạng gốc, giống file Kaggle) sang đúng định dạng model
    kỳ vọng. KHÔNG dùng pd.get_dummies() trực tiếp — với dữ liệu nhỏ/1 dòng nó drop nhầm
    category (chỉ nhìn giá trị có mặt trong chính lần gọi đó, khác lúc train trên toàn bộ
    dữ liệu). Set thủ công từng cột dummy theo đúng schema đã lưu lúc train — đã kiểm chứng
    khớp 100% với pipeline gốc trên toàn bộ 1470 dòng lẫn từng dòng đơn lẻ.

    (Bug này bắt được lần đầu khi build app.py bản Streamlit: cùng 1 nhân viên, get_dummies
    trực tiếp cho ra 4.1% trong khi pipeline gốc cho 91.5% — sai lệch hoàn toàn. Áp dụng
    luôn cách sửa đó ở đây ngay từ đầu.)
    """
    df_raw = df_raw.reset_index(drop=True)
    encoded = pd.DataFrame(0, index=df_raw.index, columns=schema["model_columns"])
    for col in schema["numeric_cols"]:
        encoded[col] = df_raw[col].values
    for idx, row in df_raw.iterrows():
        for col in schema["categorical_cols"]:
            dummy_col = f"{col}_{row[col]}"
            if dummy_col in encoded.columns:
                encoded.loc[idx, dummy_col] = 1
    encoded[schema["numeric_cols"]] = scaler.transform(encoded[schema["numeric_cols"]])
    return encoded


def risk_tier(p: float) -> str:
    if p >= THRESHOLD:
        return "🔴 Cao"
    elif p >= MEDIUM_THRESHOLD:
        return "🟡 Trung bình"
    return "🟢 Thấp"


def make_waterfall(processed_row: pd.DataFrame):
    shap_exp = explainer(processed_row)
    if len(shap_exp.shape) == 3:
        shap_exp = shap_exp[:, :, 1]
    shap_display = shap.Explanation(
        values=shap_exp.values[0],
        base_values=shap_exp.base_values[0],
        data=processed_row.iloc[0].values,
        feature_names=schema["model_columns"],
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_display, show=False, max_display=8)
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Hàm dự đoán — nhập tay 1 nhân viên
# ----------------------------------------------------------------------------
def predict_single(Age, Gender, MaritalStatus, DistanceFromHome, EducationField, Education,
                    Department, JobRole, JobLevel, MonthlyIncome, StockOptionLevel, OverTime,
                    BusinessTravel, PercentSalaryHike, PerformanceRating, JobSatisfaction,
                    EnvironmentSatisfaction, RelationshipSatisfaction, JobInvolvement,
                    WorkLifeBalance, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole,
                    YearsSinceLastPromotion, YearsWithCurrManager, NumCompaniesWorked,
                    TrainingTimesLastYear):
    row = dict(
        Age=Age, Gender=Gender, MaritalStatus=MaritalStatus, DistanceFromHome=DistanceFromHome,
        EducationField=EducationField, Education=Education, Department=Department, JobRole=JobRole,
        JobLevel=JobLevel, MonthlyIncome=MonthlyIncome, StockOptionLevel=StockOptionLevel,
        OverTime=OverTime, BusinessTravel=BusinessTravel, PercentSalaryHike=PercentSalaryHike,
        PerformanceRating=PerformanceRating, JobSatisfaction=JobSatisfaction,
        EnvironmentSatisfaction=EnvironmentSatisfaction, RelationshipSatisfaction=RelationshipSatisfaction,
        JobInvolvement=JobInvolvement, WorkLifeBalance=WorkLifeBalance, TotalWorkingYears=TotalWorkingYears,
        YearsAtCompany=YearsAtCompany, YearsInCurrentRole=YearsInCurrentRole,
        YearsSinceLastPromotion=YearsSinceLastPromotion, YearsWithCurrManager=YearsWithCurrManager,
        NumCompaniesWorked=NumCompaniesWorked, TrainingTimesLastYear=TrainingTimesLastYear,
    )
    for c in REMAINING_NUMERIC_COLS:
        row[c] = schema["numeric_ranges"][c][2]

    input_df = pd.DataFrame([row])
    processed = preprocess_input(input_df)
    proba = float(model.predict_proba(processed)[:, 1][0])
    tier = risk_tier(proba)
    fig = make_waterfall(processed)

    return f"{proba:.1%}", tier, fig


# ----------------------------------------------------------------------------
# Hàm dự đoán — batch CSV
# ----------------------------------------------------------------------------
def predict_batch(file):
    if file is None:
        return None, None
    raw_upload = pd.read_csv(file.name)
    drop_cols = [c for c in ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber", "Attrition"]
                 if c in raw_upload.columns]
    raw_clean = raw_upload.drop(columns=drop_cols)

    missing = set(schema["numeric_cols"] + schema["categorical_cols"]) - set(raw_clean.columns)
    if missing:
        raise gr.Error(f"File thiếu cột bắt buộc: {sorted(missing)}")

    processed = preprocess_input(raw_clean)
    proba = model.predict_proba(processed)[:, 1]

    result = raw_upload.copy()
    result["Risk_Score"] = proba.round(4)
    result["Risk_Level"] = [risk_tier(p) for p in proba]
    result = result.sort_values("Risk_Score", ascending=False)

    out_path = "risk_scores_output.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result, out_path


# ----------------------------------------------------------------------------
# Giao diện
# ----------------------------------------------------------------------------
def num_slider(label, col, step=1):
    lo, hi, med = schema["numeric_ranges"][col]
    return gr.Slider(minimum=lo, maximum=hi, value=med, step=step, label=label)


def cat_dropdown(label, col):
    return gr.Dropdown(choices=schema["categorical_values"][col],
                        value=schema["categorical_values"][col][0], label=label)


with gr.Blocks(title="Dự đoán Nghỉ việc Nhân sự") as demo:
    gr.Markdown(
        f"# 🔮 Hệ thống Cảnh báo Sớm Nghỉ việc Nhân sự\n"
        f"Mô hình: **{MODEL_NAME}** · Ngưỡng quyết định = **{THRESHOLD:.3f}** "
        f"(tối ưu cho Recall ≥ 0.80 — ưu tiên không bỏ sót nhân sự có nguy cơ)"
    )

    with gr.Tab("✍️ Nhập tay 1 nhân viên"):
        with gr.Group():
            gr.Markdown("### 👤 Nhân khẩu học & Cá nhân")
            with gr.Row():
                Age = num_slider("Tuổi", "Age")
                Gender = cat_dropdown("Giới tính", "Gender")
                MaritalStatus = cat_dropdown("Tình trạng hôn nhân", "MaritalStatus")
            with gr.Row():
                DistanceFromHome = num_slider("Khoảng cách nhà-cty (km)", "DistanceFromHome")
                EducationField = cat_dropdown("Ngành học", "EducationField")
                Education = num_slider("Trình độ học vấn (1-5)", "Education")

        with gr.Group():
            gr.Markdown("### 💼 Công việc & Thù lao")
            with gr.Row():
                Department = cat_dropdown("Phòng ban", "Department")
                JobRole = cat_dropdown("Vị trí công việc", "JobRole")
                JobLevel = num_slider("Cấp bậc (1-5)", "JobLevel")
            with gr.Row():
                MonthlyIncome = num_slider("Lương tháng ($)", "MonthlyIncome", step=100)
                StockOptionLevel = num_slider("Mức cổ phần thưởng (0-3)", "StockOptionLevel")
                OverTime = cat_dropdown("Làm thêm giờ", "OverTime")
            with gr.Row():
                BusinessTravel = cat_dropdown("Tần suất công tác", "BusinessTravel")
                PercentSalaryHike = num_slider("% tăng lương gần nhất", "PercentSalaryHike")
                PerformanceRating = num_slider("Đánh giá hiệu suất (1-4)", "PerformanceRating")

        with gr.Group():
            gr.Markdown("### 😊 Mức độ hài lòng")
            with gr.Row():
                JobSatisfaction = num_slider("Hài lòng công việc (1-4)", "JobSatisfaction")
                EnvironmentSatisfaction = num_slider("Hài lòng môi trường (1-4)", "EnvironmentSatisfaction")
                RelationshipSatisfaction = num_slider("Hài lòng đồng nghiệp (1-4)", "RelationshipSatisfaction")
            with gr.Row():
                JobInvolvement = num_slider("Mức độ gắn kết (1-4)", "JobInvolvement")
                WorkLifeBalance = num_slider("Cân bằng công việc-cuộc sống (1-4)", "WorkLifeBalance")

        with gr.Group():
            gr.Markdown("### 📅 Lịch sử công tác")
            with gr.Row():
                TotalWorkingYears = num_slider("Tổng số năm đi làm", "TotalWorkingYears")
                YearsAtCompany = num_slider("Số năm tại công ty", "YearsAtCompany")
                YearsInCurrentRole = num_slider("Số năm ở vị trí hiện tại", "YearsInCurrentRole")
            with gr.Row():
                YearsSinceLastPromotion = num_slider("Số năm từ lần thăng chức cuối", "YearsSinceLastPromotion")
                YearsWithCurrManager = num_slider("Số năm với quản lý hiện tại", "YearsWithCurrManager")
            with gr.Row():
                NumCompaniesWorked = num_slider("Số công ty đã từng làm", "NumCompaniesWorked")
                TrainingTimesLastYear = num_slider("Số lần đào tạo năm ngoái", "TrainingTimesLastYear")

        predict_btn = gr.Button("🔮 Dự đoán", variant="primary", size="lg")

        with gr.Row():
            proba_out = gr.Textbox(label="Xác suất nghỉ việc")
            tier_out = gr.Textbox(label="Mức rủi ro")
        shap_out = gr.Plot(label="Vì sao mô hình dự đoán như vậy (SHAP)")

        predict_btn.click(
            predict_single,
            inputs=[Age, Gender, MaritalStatus, DistanceFromHome, EducationField, Education,
                    Department, JobRole, JobLevel, MonthlyIncome, StockOptionLevel, OverTime,
                    BusinessTravel, PercentSalaryHike, PerformanceRating, JobSatisfaction,
                    EnvironmentSatisfaction, RelationshipSatisfaction, JobInvolvement,
                    WorkLifeBalance, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole,
                    YearsSinceLastPromotion, YearsWithCurrManager, NumCompaniesWorked,
                    TrainingTimesLastYear],
            outputs=[proba_out, tier_out, shap_out],
        )

    with gr.Tab("📁 Tải CSV hàng loạt"):
        gr.Markdown(
            "Tải lên CSV theo **đúng định dạng gốc** (giống file IBM HR trên Kaggle, "
            "không cần encode/scale trước — app tự xử lý)."
        )
        file_in = gr.File(label="Chọn file CSV", file_types=[".csv"])
        batch_btn = gr.Button("Chấm điểm rủi ro", variant="primary")
        result_df = gr.Dataframe(label="Kết quả")
        result_file = gr.File(label="Tải kết quả CSV")

        batch_btn.click(predict_batch, inputs=file_in, outputs=[result_df, result_file])

    gr.Markdown(
        "---\n⚠️ Công cụ hỗ trợ ra quyết định — không thay thế đánh giá của chuyên viên HR. "
        "Dữ liệu: IBM HR Analytics Employee Attrition & Performance (Kaggle)."
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860)
