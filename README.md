# 🔮 Dự Đoán & Cảnh Báo Sớm Nghỉ Việc Nhân Sự

Hệ thống Machine Learning dự đoán nguy cơ nghỉ việc của nhân viên, giúp bộ phận HR can
thiệp giữ chân nhân tài sớm thay vì bị động — xây dựng theo lộ trình 4 ngày full-time,
từ EDA đến mô hình đã tune, giải thích được (SHAP), và dashboard demo được.

🔗 **Demo trực tiếp:** _(dán link Streamlit Community Cloud sau khi deploy — xem hướng dẫn bên dưới)_

## 📌 Bài toán

Chi phí thay thế một nhân viên thường được ước tính trong khoảng 50–200% lương năm của vị
trí đó (tuyển dụng, đào tạo lại, năng suất sụt giảm, mất tri thức ngầm). Dự án này xây một
mô hình dự đoán **trước** ai có khả năng nghỉ việc cao, thay vì chỉ phân tích nguyên nhân
**sau khi** họ đã rời đi — để HR có thể hành động sớm và cá nhân hoá.

## 📊 Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
— 1.470 nhân viên, 35 thuộc tính, dữ liệu tổng hợp (synthetic) do IBM phát hành. Biến mục
tiêu `Attrition` mất cân bằng: 83.9% No / 16.1% Yes.

## 🧪 Phương pháp

1. **EDA + tiền xử lý:** loại cột vô nghĩa, One-Hot Encoding, StandardScaler
2. **Stratified Train/Test Split** (80/20) — bảo toàn tỷ lệ lớp
3. **SMOTE** xử lý mất cân bằng — chỉ áp dụng trên tập train, đóng gói trong
   `imblearn.Pipeline` để tránh rò rỉ dữ liệu giữa các fold khi cross-validate
4. **So sánh 5 mô hình:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
   (Stratified 5-Fold CV)
5. **Tune sâu** 2 ứng viên tốt nhất bằng `RandomizedSearchCV` (tối ưu theo PR-AUC)
6. **Threshold Tuning** theo chi phí kinh doanh — ưu tiên Recall ≥ 0.80 vì bỏ sót một
   nhân sự sắp nghỉ (False Negative) tốn kém hơn nhiều so với báo động nhầm (False Positive)
7. **Explainable AI** với SHAP (TreeExplainer) — beeswarm (global) + waterfall (local)

## 📈 Kết quả

| Mô hình | Ngưỡng | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.50 | 0.56 | 0.49 | 0.52 | – | – |
| Logistic Regression (đã tune) | 0.50 | 0.356 | 0.681 | 0.467 | 0.568 | 0.786 |
| CatBoost (đã tune) | 0.50 | 0.778 | 0.298 | 0.431 | 0.572 | 0.801 |
| **CatBoost (đã tune) — MÔ HÌNH CUỐI** | **0.092** | **0.342** | **0.809** | **0.481** | **0.572** | **0.801** |

CatBoost thắng về PR-AUC/ROC-AUC (thước đo không phụ thuộc ngưỡng). Ở ngưỡng mặc định 0.5
nó có Precision rất cao nhưng Recall quá thấp (chỉ bắt được 30% người sắp nghỉ) — sau khi
hạ ngưỡng xuống 0.092 theo đúng yêu cầu kinh doanh, mô hình bắt được **81%** người sắp nghỉ
việc, đánh đổi bằng việc chấp nhận báo động nhầm nhiều hơn (Precision 34%).

**Trên tập test (294 người):** bắt đúng 38/47 người sắp nghỉ thật (bỏ sót 9), báo nhầm
73/247 người không có ý định nghỉ.

## 🔑 Insight chính (SHAP)

`OverTime` (làm thêm giờ) là yếu tố ảnh hưởng mạnh nhất đến rủi ro nghỉ việc — bỏ xa các
yếu tố còn lại. Theo sau là `BusinessTravel` (đi công tác thường xuyên) và `StockOptionLevel`
thấp. Hai nhóm có rủi ro nội tại cao hơn hẳn: phòng **Sales** và vị trí **Laboratory
Technician**. Đáng chú ý: `MonthlyIncome` — dù tương quan thô khá cao — lại **không** nằm
trong top 10 yếu tố SHAP quan trọng nhất, cho thấy đãi ngộ tài chính không phải đòn bẩy
mạnh nhất; chính sách làm thêm giờ và cường độ công tác mới là nơi HR nên ưu tiên rà soát.

![SHAP Beeswarm](images/10_shap_beeswarm.png)

## 🔬 Phân tích chuyên sâu (`notebooks/05_deep_dive_insights.ipynb`)

Ngoài quy trình 4 ngày chuẩn, có thêm 1 notebook đào sâu vào những câu hỏi mà một mô hình
"chạy đúng số liệu" chưa tự trả lời được:

- **Chi phí thực ($):** tối ưu chi phí không ràng buộc luôn đẩy ngưỡng về gần 0 (flag gần
  hết công ty) — vô dụng trên thực tế vì HR không đủ năng lực can thiệp cá nhân hoá hàng
  trăm người. Thêm ràng buộc theo năng lực HR (capacity) mới ra được khuyến nghị hành động
  được: ở ngưỡng đang dùng (~37% capacity), ước tính tiết kiệm hàng trăm nghìn $ so với
  không dùng model, trên quy mô tập test.
- **Mô hình bỏ sót ai:** nhóm bị bỏ sót (False Negative) có thu nhập cao gấp ~2.4 lần, thâm
  niên gấp ~4.4 lần nhóm bắt đúng, và **0% làm thêm giờ** (so với 79% ở nhóm bắt đúng) — mô
  hình học rất tốt kiểu "trẻ, OT nhiều, dễ burnout" nhưng gần như mù với kiểu "nghỉ việc âm
  thầm" của nhân sự thâm niên, thu nhập tốt. Đây là giới hạn thật, cần thêm dữ liệu mới
  (khảo sát gắn kết, tín hiệu thị trường lao động) mới thu hẹp được, không thể tune ra.
- **Công bằng thuật toán:** ổn theo giới tính (Disparate Impact Ratio 0.899, đạt quy tắc
  4/5); có tín hiệu đáng theo dõi ở nhóm tuổi 45+ (Recall chỉ 0.667, thấp hơn hẳn 2 nhóm còn
  lại) — khớp với phát hiện ở trên, dù cỡ mẫu còn nhỏ (9 người).
- **Driver theo phòng ban:** bất ngờ là top 3 yếu tố quan trọng nhất **giống hệt nhau** ở cả
  3 phòng ban — nghĩa là chính sách rà soát OT/công tác nên áp dụng company-wide thay vì
  thiết kế riêng theo từng phòng.
- **4 chân dung người nghỉ việc** (`06_personas_and_baseline.ipynb`, K-Means trên toàn bộ 237
  người đã nghỉ): không chỉ 1 kiểu người rời đi. Nổi bật nhất — cụm "rời đi trong âm thầm"
  (17 người, lớn tuổi nhất, lương cao nhất, thâm niên 20+ năm, hầu như không OT) **trùng khớp
  độc lập lần 2** với nhóm False Negative ở trên, bằng phương pháp hoàn toàn khác (clustering
  không giám sát thay vì phân tích lỗi có giám sát) — 2 phương pháp độc lập cùng kết luận là
  tín hiệu đáng tin hơn nhiều so với 1 phân tích đơn lẻ.
- **ML có đáng công sức không?** So với luật đơn giản "ai đang OT thì liên hệ", ở cùng mức độ
  bao phủ, ML chỉ nhỉnh hơn +2 điểm % Recall — khá khiêm tốn. Khoảng cách nới rộng hơn
  (+10 điểm %) khi so với luật 2 biến. Kết luận trung thực: giá trị thật của ML không nằm ở
  quyết định nhị phân "liên hệ hay không", mà ở khả năng **xếp hạng liên tục** theo capacity
  thực tế, **kết hợp nhiều tín hiệu yếu**, và **giải thích được** — 3 thứ luật if-else không làm được.

## 🖥️ Dashboard demo

Dự án có **2 phiên bản dashboard** (cùng logic, khác nền tảng) để demo:

| | Streamlit | Gradio |
|---|---|---|
| File | `app/app.py` | `app/gradio_app.py` |
| Chạy local | `streamlit run app/app.py` | `python app/gradio_app.py` |
| Deploy free | Streamlit Community Cloud | Hugging Face Spaces |
| Phù hợp khi | dashboard nội bộ, nhiều panel | demo model chia sẻ nhanh, hệ sinh thái AI/HuggingFace |

Cả 2 đều có **3 chế độ**:
- **Tải CSV hàng loạt** (định dạng gốc như file Kaggle) → chấm điểm rủi ro toàn bộ danh sách
- **Nhập tay 1 nhân viên** → dự đoán tức thì + giải thích SHAP + **what-if simulator**: thử
  đổi OverTime/StockOptionLevel/BusinessTravel và xem rủi ro thay đổi ngay lập tức — dựa trên
  3 đòn bẩy HR thực sự can thiệp được (notebook 05, 06)
- **🎯 Top rủi ro cao nhất**: chấm điểm toàn bộ nhân sự hiện có (không cần upload), xếp hạng
  Top N kèm lý do chính (SHAP) — sẵn sàng dùng để triage ngay, không cần chuẩn bị dữ liệu gì thêm

### Deploy Gradio lên Hugging Face Spaces (free)

1. Tạo tài khoản tại huggingface.co → **New Space** → chọn SDK = **Gradio**
2. Trong Space mới, tải lên: `app/gradio_app.py` (đổi tên thành `app.py`), toàn bộ thư mục
   `models/`, và `requirements.txt`
3. Sửa dòng cuối `gradio_app.py` nếu cần: Spaces tự chạy file `app.py`, không cần gọi `demo.launch()`
   thủ công (Spaces tự nhận biến `demo`)
4. Space build xong sẽ có link dạng `https://huggingface.co/spaces/<username>/<space-name>` — dán vào đầu README

## 🚀 Cách chạy dự án

```bash
git clone https://github.com/<username>/employee-attrition-prediction.git
cd employee-attrition-prediction
pip install -r requirements.txt

# Mở các notebook theo thứ tự 01 -> 04 (thư mục notebooks/), hoặc chạy thẳng dashboard:
streamlit run app/app.py        # bản Streamlit
python app/gradio_app.py        # bản Gradio (http://localhost:7860)
```

## 📁 Cấu trúc thư mục

```
employee-attrition-prediction/
├── data/
│   ├── raw/                          # dataset gốc từ Kaggle
│   └── processed/                    # dữ liệu đã encode, kết quả so sánh mô hình, risk segmentation
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_split_smote_baseline.ipynb
│   ├── 03_tuning_evaluation_threshold.ipynb
│   ├── 04_shap_explainability.ipynb
│   ├── 05_deep_dive_insights.ipynb   # cost-benefit, error analysis, fairness audit
│   └── 06_personas_and_baseline.ipynb # persona clustering, ML vs luật đơn giản
├── models/                           # scaler, model cuối, threshold, schema cho app
├── images/                           # toàn bộ biểu đồ (EDA, SHAP, evaluation)
├── app/
│   ├── app.py                        # Streamlit dashboard
│   └── gradio_app.py                 # Gradio dashboard (deploy free lên HuggingFace Spaces)
├── requirements.txt
└── README.md
```

## 🐛 Những lỗi thật gặp phải trong lúc làm

Không phải mọi thứ đúng ngay từ lần chạy đầu — 3 lỗi dưới đây là thật, bắt được khi số liệu
trông "đẹp bất thường" hoặc sai rõ ràng, không phải liệt kê cho có. Chi tiết trước/sau nằm
trong git log (`git log --oneline`) và trong chính các notebook, tại đúng chỗ lỗi xảy ra.

1. **SMOTE rò rỉ qua các fold Cross-Validation** (Ngày 2) — so sánh nhanh 5 mô hình bằng
   cách đưa thẳng dữ liệu đã SMOTE vào `cross_validate()`. Recall/Precision/PR-AUC của
   *mọi* mô hình đều ra 90-98%, cao hơn baseline (~50%) một cách vô lý. Nguyên nhân: một
   điểm tổng hợp ở fold validation có thể được nội suy từ điểm gốc đang nằm ở fold train
   của chính vòng đó. Sửa bằng `imblearn.Pipeline` để SMOTE chạy lại trong từng fold — số
   liệu rơi về mức thực tế hơn (Recall 34-49%).
2. **Logic tìm threshold bị đảo ngược** (Ngày 3) — dùng `np.argmax(recalls >= 0.80)` để
   tìm ngưỡng đạt Recall≥0.80, nhưng `argmax` trả về vị trí *đầu tiên* thoả điều kiện, trong
   khi recall giảm dần theo threshold tăng → chọn nhầm ngưỡng ≈0.001, model gắn cờ gần hết
   mọi người là rủi ro cao (Recall=1.0 nhưng Precision=0.16, vô dụng). Sửa bằng cách lấy vị
   trí *cuối cùng* thoả điều kiện thay vì đầu tiên.
3. **`pd.get_dummies(drop_first=True)` vỡ khi encode 1 dòng dữ liệu** (Ngày 4, lúc build
   dashboard) — hàm tiền xử lý cho form nhập tay gọi `get_dummies` trực tiếp trên 1 dòng,
   nhưng với 1 dòng thì mỗi cột categorical chỉ có 1 giá trị, nên `drop_first` xoá luôn giá
   trị đó bất kể nó có phải category tham chiếu lúc train hay không. Cùng 1 nhân viên, xác
   suất đúng là 91.5% nhưng hàm lỗi cho ra 4.1%. Sửa bằng cách encode thủ công theo đúng
   schema cột đã lưu, verify lại khớp 100% trên toàn bộ 1470 dòng lẫn từng dòng đơn lẻ.

## 🔄 Nếu làm lại, mình sẽ cải thiện thêm

- **Hiệu chỉnh xác suất (calibration):** threshold cuối cùng chỉ 0.092 — dấu hiệu xác suất
  CatBoost đang lệch thấp so với xác suất "đúng nghĩa". Platt scaling hoặc Isotonic
  Regression (`CalibratedClassifierCV`) có thể giúp con số dễ diễn giải hơn cho HR.
- **Ngưỡng phân khúc Trung bình (threshold/3) là heuristic mình tự đặt**, chưa được ai
  ngoài kiểm chứng — nên ngồi lại với HR thật để xem 3 mức Thấp/Trung bình/Cao có khớp với
  cách họ muốn hành động hay không, thay vì tự quyết một mình.
- **73/247 false positives ở tập test là chi phí vận hành thật** (gần 30% người bị gắn cờ
  nhầm) — đáng lẽ nên trình bày rõ đánh đổi này với stakeholder trước khi chốt threshold,
  chứ không chỉ tối ưu Recall một chiều.
- **Chỉ có 1 snapshot dữ liệu**, chưa kiểm tra model có "trôi" (drift) theo thời gian khi
  đặc điểm nhân sự công ty thay đổi — nếu triển khai thật cần theo dõi định kỳ.

## 🛠️ Tech Stack

Python · pandas · scikit-learn · imbalanced-learn · CatBoost/XGBoost/LightGBM · SHAP · Streamlit · Gradio

## 👤 Tác giả

_[Tên bạn]_ — _[LinkedIn/GitHub của bạn]_
