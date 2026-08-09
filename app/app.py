"""
Dashboard Dự đoán & Cảnh báo Sớm Nghỉ Việc Nhân Sự
---------------------------------------------------
Chạy: streamlit run app/app.py  (từ thư mục gốc project)
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dự đoán Nghỉ việc Nhân sự", page_icon="🔮", layout="wide")


# ----------------------------------------------------------------------------
# Load model, scaler, schema (cache để không load lại mỗi lần tương tác)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/threshold.txt") as f:
        threshold = float(f.read())
    with open("models/model_name.txt") as f:
        model_name = f.read()
    with open("models/feature_schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    return model, scaler, threshold, model_name, schema, explainer


model, scaler, THRESHOLD, MODEL_NAME, schema, explainer = load_artifacts()
MEDIUM_THRESHOLD = THRESHOLD / 3


def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Chuyển dữ liệu THÔ (định dạng gốc, giống file Kaggle) sang đúng định dạng model
    kỳ vọng. KHÔNG dùng pd.get_dummies() trực tiếp ở đây — với dữ liệu nhỏ/1 dòng nó sẽ
    drop nhầm category (vì drop_first chỉ nhìn giá trị có mặt trong chính lần gọi đó, khác
    hẳn lúc train trên toàn bộ dữ liệu). Set thủ công từng cột dummy theo đúng schema đã lưu.

    Bug thật gặp phải khi build hàm này: bản đầu gọi thẳng get_dummies(drop_first=True)
    trên 1 dòng input từ form. Với 1 dòng, mỗi cột categorical chỉ có 1 giá trị -> drop_first
    xoá luôn giá trị đó, bất kể nó có phải category tham chiếu lúc train hay không. Cùng một
    nhân viên: pipeline gốc cho xác suất 91.5%, bản lỗi cho ra 4.1%. Đã sửa + verify khớp
    100% trên toàn bộ 1470 dòng lẫn từng dòng đơn lẻ trước khi dùng thật.
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
        return "Cao"
    elif p >= MEDIUM_THRESHOLD:
        return "Trung bình"
    return "Thấp"


TIER_COLOR = {"Thấp": "🟢", "Trung bình": "🟡", "Cao": "🔴"}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🔮 Hệ thống Cảnh báo Sớm Nghỉ việc Nhân sự")
st.caption(
    f"Mô hình: **{MODEL_NAME}** · Ngưỡng quyết định = **{THRESHOLD:.3f}** "
    f"(đã tối ưu cho Recall ≥ 0.80 — ưu tiên không bỏ sót nhân sự có nguy cơ)"
)

tab1, tab2, tab3 = st.tabs(["📁 Tải CSV hàng loạt", "✍️ Nhập tay 1 nhân viên", "🎯 Top rủi ro cao nhất"])

# ----------------------------------------------------------------------------
# TAB 1 — Batch CSV
# ----------------------------------------------------------------------------
with tab1:
    st.write(
        "Tải lên CSV theo **đúng định dạng gốc** (giống file IBM HR trên Kaggle, "
        "không cần encode/scale trước — app tự xử lý)."
    )
    uploaded = st.file_uploader("Chọn file CSV", type="csv")

    if uploaded:
        try:
            raw_upload = pd.read_csv(uploaded)
            drop_cols = [c for c in ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber", "Attrition"]
                         if c in raw_upload.columns]
            raw_clean = raw_upload.drop(columns=drop_cols)

            missing = set(schema["numeric_cols"] + schema["categorical_cols"]) - set(raw_clean.columns)
            if missing:
                st.error(f"File thiếu cột bắt buộc: {sorted(missing)}")
            else:
                processed = preprocess_input(raw_clean)
                proba = model.predict_proba(processed)[:, 1]

                result = raw_upload.copy()
                result["Risk_Score"] = proba.round(4)
                result["Risk_Level"] = [risk_tier(p) for p in proba]
                result = result.sort_values("Risk_Score", ascending=False)

                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Nguy cơ Cao", int((result["Risk_Level"] == "Cao").sum()))
                c2.metric("🟡 Nguy cơ Trung bình", int((result["Risk_Level"] == "Trung bình").sum()))
                c3.metric("🟢 Nguy cơ Thấp", int((result["Risk_Level"] == "Thấp").sum()))

                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "⬇️ Tải kết quả CSV", result.to_csv(index=False).encode("utf-8-sig"),
                    "risk_scores.csv", "text/csv",
                )
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")
    else:
        st.info("Chưa có file nào được tải lên. Có thể dùng thử file mẫu: "
                "`data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`.")

# ----------------------------------------------------------------------------
# TAB 2 — Nhập tay 1 nhân viên
# ----------------------------------------------------------------------------
with tab2:
    st.write("Nhập thông tin nhân viên để dự đoán rủi ro nghỉ việc ngay lập tức.")

    def num_input(label, col, step=1):
        lo, hi, med = schema["numeric_ranges"][col]
        return st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(med), step=step)

    def cat_input(label, col):
        return st.selectbox(label, schema["categorical_values"][col])

    with st.form("employee_form"):
        st.subheader("👤 Nhân khẩu học & Cá nhân")
        c1, c2, c3 = st.columns(3)
        with c1:
            Age = num_input("Tuổi", "Age")
            Gender = cat_input("Giới tính", "Gender")
        with c2:
            MaritalStatus = cat_input("Tình trạng hôn nhân", "MaritalStatus")
            DistanceFromHome = num_input("Khoảng cách nhà-cty (km)", "DistanceFromHome")
        with c3:
            EducationField = cat_input("Ngành học", "EducationField")
            Education = num_input("Trình độ học vấn (1-5)", "Education")

        st.subheader("💼 Công việc & Thù lao")
        c1, c2, c3 = st.columns(3)
        with c1:
            Department = cat_input("Phòng ban", "Department")
            JobRole = cat_input("Vị trí công việc", "JobRole")
            JobLevel = num_input("Cấp bậc (1-5)", "JobLevel")
        with c2:
            MonthlyIncome = num_input("Lương tháng ($)", "MonthlyIncome", step=100)
            StockOptionLevel = num_input("Mức cổ phần thưởng (0-3)", "StockOptionLevel")
            OverTime = cat_input("Làm thêm giờ", "OverTime")
        with c3:
            BusinessTravel = cat_input("Tần suất công tác", "BusinessTravel")
            PercentSalaryHike = num_input("% tăng lương gần nhất", "PercentSalaryHike")
            PerformanceRating = num_input("Đánh giá hiệu suất (1-4)", "PerformanceRating")

        st.subheader("😊 Mức độ hài lòng")
        c1, c2, c3 = st.columns(3)
        with c1:
            JobSatisfaction = num_input("Hài lòng công việc (1-4)", "JobSatisfaction")
            EnvironmentSatisfaction = num_input("Hài lòng môi trường (1-4)", "EnvironmentSatisfaction")
        with c2:
            RelationshipSatisfaction = num_input("Hài lòng quan hệ đồng nghiệp (1-4)", "RelationshipSatisfaction")
            JobInvolvement = num_input("Mức độ gắn kết công việc (1-4)", "JobInvolvement")
        with c3:
            WorkLifeBalance = num_input("Cân bằng công việc-cuộc sống (1-4)", "WorkLifeBalance")

        st.subheader("📅 Lịch sử công tác")
        c1, c2, c3 = st.columns(3)
        with c1:
            TotalWorkingYears = num_input("Tổng số năm đi làm", "TotalWorkingYears")
            YearsAtCompany = num_input("Số năm tại công ty", "YearsAtCompany")
        with c2:
            YearsInCurrentRole = num_input("Số năm ở vị trí hiện tại", "YearsInCurrentRole")
            YearsSinceLastPromotion = num_input("Số năm từ lần thăng chức cuối", "YearsSinceLastPromotion")
        with c3:
            YearsWithCurrManager = num_input("Số năm với quản lý hiện tại", "YearsWithCurrManager")
            NumCompaniesWorked = num_input("Số công ty đã từng làm", "NumCompaniesWorked")
            TrainingTimesLastYear = num_input("Số lần đào tạo năm ngoái", "TrainingTimesLastYear")

        # Các cột numeric còn lại trong schema nhưng ít khi HR quan tâm nhập tay — điền giá trị trung vị mặc định
        remaining = [c for c in schema["numeric_cols"] if c not in [
            "Age", "DistanceFromHome", "Education", "JobLevel", "MonthlyIncome",
            "StockOptionLevel", "PercentSalaryHike", "PerformanceRating", "JobSatisfaction",
            "EnvironmentSatisfaction", "RelationshipSatisfaction", "JobInvolvement",
            "WorkLifeBalance", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
            "TrainingTimesLastYear",
        ]]

        submitted = st.form_submit_button("🔮 Dự đoán", use_container_width=True)

    if submitted:
        row = {
            "Age": Age, "Gender": Gender, "MaritalStatus": MaritalStatus,
            "DistanceFromHome": DistanceFromHome, "EducationField": EducationField, "Education": Education,
            "Department": Department, "JobRole": JobRole, "JobLevel": JobLevel,
            "MonthlyIncome": MonthlyIncome, "StockOptionLevel": StockOptionLevel, "OverTime": OverTime,
            "BusinessTravel": BusinessTravel, "PercentSalaryHike": PercentSalaryHike,
            "PerformanceRating": PerformanceRating, "JobSatisfaction": JobSatisfaction,
            "EnvironmentSatisfaction": EnvironmentSatisfaction,
            "RelationshipSatisfaction": RelationshipSatisfaction, "JobInvolvement": JobInvolvement,
            "WorkLifeBalance": WorkLifeBalance, "TotalWorkingYears": TotalWorkingYears,
            "YearsAtCompany": YearsAtCompany, "YearsInCurrentRole": YearsInCurrentRole,
            "YearsSinceLastPromotion": YearsSinceLastPromotion,
            "YearsWithCurrManager": YearsWithCurrManager, "NumCompaniesWorked": NumCompaniesWorked,
            "TrainingTimesLastYear": TrainingTimesLastYear,
        }
        for c in remaining:
            row[c] = schema["numeric_ranges"][c][2]
        st.session_state["last_row"] = row  # lưu lại để phần what-if bên dưới dùng, không cần submit lại

    if "last_row" in st.session_state:
        row = st.session_state["last_row"]
        input_df = pd.DataFrame([row])
        processed = preprocess_input(input_df)
        proba = float(model.predict_proba(processed)[:, 1][0])
        tier = risk_tier(proba)

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Xác suất nghỉ việc", f"{proba:.1%}")
            st.markdown(f"### {TIER_COLOR[tier]} Mức rủi ro: **{tier}**")
            st.progress(min(proba, 1.0))

        with r2:
            st.write("**Vì sao mô hình dự đoán như vậy** (SHAP — 8 yếu tố ảnh hưởng nhiều nhất):")
            shap_exp = explainer(processed)
            if len(shap_exp.shape) == 3:
                shap_exp = shap_exp[:, :, 1]
            # Gắn lại dữ liệu đã encode (đủ 44 cột, đúng feature_names) để waterfall hiển thị đúng
            shap_display = shap.Explanation(
                values=shap_exp.values[0],
                base_values=shap_exp.base_values[0],
                data=processed.iloc[0].values,
                feature_names=schema["model_columns"],
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_display, show=False, max_display=8)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.divider()
        st.write("#### 🔄 Thử what-if: nếu đổi 1 yếu tố, rủi ro thay đổi thế nào?")
        st.caption(
            "Dựa trên phát hiện ở notebook phân tích chuyên sâu (05, 06): OverTime là yếu tố "
            "mạnh nhất, StockOptionLevel và BusinessTravel cũng ảnh hưởng rõ rệt. Đây là những "
            "đòn bẩy HR có thể thực sự can thiệp được — không phải mọi yếu tố đều vậy (VD: "
            "không thể 'đổi Age')."
        )
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            wi_overtime = st.selectbox("Nếu OverTime đổi thành", schema["categorical_values"]["OverTime"],
                                        index=schema["categorical_values"]["OverTime"].index(row["OverTime"]),
                                        key="wi_ot")
        with wcol2:
            wi_stock = st.slider("Nếu StockOptionLevel đổi thành", 0, 3, int(row["StockOptionLevel"]), key="wi_stock")
        with wcol3:
            wi_travel = st.selectbox("Nếu BusinessTravel đổi thành", schema["categorical_values"]["BusinessTravel"],
                                      index=schema["categorical_values"]["BusinessTravel"].index(row["BusinessTravel"]),
                                      key="wi_travel")

        wi_row = dict(row)
        wi_row["OverTime"] = wi_overtime
        wi_row["StockOptionLevel"] = wi_stock
        wi_row["BusinessTravel"] = wi_travel
        wi_processed = preprocess_input(pd.DataFrame([wi_row]))
        wi_proba = float(model.predict_proba(wi_processed)[:, 1][0])
        delta = wi_proba - proba

        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Rủi ro hiện tại", f"{proba:.1%}")
        wc2.metric("Rủi ro sau khi đổi", f"{wi_proba:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
        wc3.metric("Mức rủi ro mới", risk_tier(wi_proba))

with tab3:
    st.write(
        "Chấm điểm rủi ro cho **toàn bộ nhân sự hiện có** (dùng luôn dataset đã có, "
        "không cần tải file) — dùng để triage nhanh: HR nên ưu tiên liên hệ ai trước."
    )
    top_n = st.slider("Số lượng hiển thị", 5, 100, 20, step=5)

    @st.cache_data
    def compute_leaderboard():
        raw_full = pd.read_csv("data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
        drop_cols = [c for c in ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber", "Attrition"]
                     if c in raw_full.columns]
        raw_clean = raw_full.drop(columns=drop_cols)
        processed_full = preprocess_input(raw_clean)
        proba_full = model.predict_proba(processed_full)[:, 1]
        return raw_full, processed_full, proba_full

    raw_full, processed_full, proba_full = compute_leaderboard()
    order = np.argsort(-proba_full)[:top_n]

    shap_top = explainer(processed_full.iloc[order])
    if len(shap_top.shape) == 3:
        shap_top = shap_top[:, :, 1]

    reasons = []
    for i in range(len(order)):
        vals = pd.Series(shap_top.values[i], index=schema["model_columns"])
        top2 = vals[vals > 0].sort_values(ascending=False).head(2)
        reasons.append(" · ".join(top2.index.tolist()) if len(top2) else "—")

    display_cols = ["Department", "JobRole", "Age", "MonthlyIncome", "OverTime", "YearsAtCompany"]
    leaderboard = raw_full.iloc[order][display_cols].copy()
    leaderboard.insert(0, "Hạng", range(1, len(order) + 1))
    leaderboard["Xác suất nghỉ việc"] = [f"{p:.1%}" for p in proba_full[order]]
    leaderboard["Mức rủi ro"] = [risk_tier(p) for p in proba_full[order]]
    leaderboard["Lý do chính (SHAP)"] = reasons

    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Tải leaderboard đầy đủ",
        leaderboard.to_csv(index=False).encode("utf-8-sig"),
        "top_risk_leaderboard.csv", "text/csv",
    )

st.divider()
st.caption(
    "⚠️ Công cụ hỗ trợ ra quyết định — không thay thế đánh giá của chuyên viên HR. "
    "Dữ liệu: IBM HR Analytics Employee Attrition & Performance (Kaggle)."
)
