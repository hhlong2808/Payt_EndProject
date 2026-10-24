# Dự Án Dự Đoán Giá Điện Thoại Theo Thông Số Kỹ Thuật (Smartphone Price Prediction)

Dự án xây dựng mô hình Học máy (Machine Learning) nhằm ước lượng giá bán (`approx_price_EUR`) của điện thoại thông minh dựa trên các thông số phần cứng, công nghệ màn hình, kết nối và thời điểm ra mắt từ bộ dữ liệu GSMArena.

---

## 1. Cấu trúc thư mục

```text
Testing model/
├── data/
│   ├── phone_dataset .csv             # Dữ liệu gốc ban đầu (8.628 thiết bị, 40 cột thô)
│   ├── smartphones_train_clean.csv    # Tập huấn luyện đã xử lý khuyết thiếu (70% - 1.337 mẫu)
│   ├── smartphones_val_clean.csv      # Tập kiểm thử quá trình (15% - 286 mẫu)
│   └── smartphones_test_clean.csv     # Tập kiểm thử cuối cùng (15% - 287 mẫu)
├── phone_dataset_cleaned.csv          # Bảng dữ liệu smartphone sau khi lọc & tách Regex (1.910 mẫu, 21 cột)
├── Data_cleaning.ipynb                # Bước 1: Làm sạch dữ liệu thô & Bóc tách thông số bằng Regex
├── analysis.ipynb                     # Bước 2: Phân tích EDA, Chia tập dữ liệu, Imputation & Baseline Model
├── Linear_model.ipynb                 # Bước 3: Feature Engineering & Tối ưu mô hình Tuyến tính / ElasticNet
├── RandomForest&XGBoost.ipynb         # Bước 4: Target Encoding, Tinh chỉnh Random Forest, XGBoost & Ensemble
└── README.md                          # Tài liệu hướng dẫn và tổng kết dự án
```

---

## 2. Cài đặt thư viện

Yêu cầu môi trường **Python 3.9+**. Mở Terminal / PowerShell tại thư mục dự án và cài đặt các thư viện cần thiết:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

*(Nếu sử dụng môi trường ảo có sẵn trong thư mục dự án: `.\myenv\Scripts\activate`)*

---

## 3. Hướng dẫn chạy dự án

Chạy lần lượt các Jupyter Notebook theo đúng thứ tự pipeline sau:

1. **`Data_cleaning.ipynb`**:
   * Đọc dữ liệu gốc `data/phone_dataset .csv`.
   * Lọc bỏ máy cục gạch (Feature phone), máy tính bảng (Tablet), Smartwatch và ngoại lai xa xỉ (Vertu, giá > 2500 EUR).
   * Dùng Regex trích xuất các thuộc tính số học và phân loại, xuất ra file `phone_dataset_cleaned.csv`.
2. **`analysis.ipynb`**:
   * Khám phá phân phối dữ liệu (EDA), kiểm tra độ lệch giá bán và mối tương quan giữa phần cứng với giá.
   * Chia tập dữ liệu theo tỷ lệ **70% Train - 15% Validation - 15% Test** (có phân tầng *Stratified Split* theo phân khúc giá).
   * Điền giá trị khuyết thiếu (Median/Mode học từ tập Train) chống rò rỉ dữ liệu (Data Leakage) và xuất ra 3 file CSV trong thư mục `data/`.
   * Huấn luyện 3 mô hình Baseline làm mốc so sánh chuẩn.
3. **`Linear_model.ipynb`**:
   * Thử nghiệm từng bước kỹ thuật tạo đặc trưng (Feature Engineering): Đặc trưng vật lý (`PPI`, `Tỉ số Pin/Độ Dày`, `Tổng Xung Nhịp`, `Chỉ Số Hiệu Năng`, `Tổng MP Camera`) và đặc trưng tương đối theo dòng chảy thời gian (Định luật Moore).
   * Biến đổi Logarit (`np.log1p`), chuẩn hóa dữ liệu `StandardScaler` và giải quyết hiện tượng đa cộng tuyến bằng mô hình điều chuẩn **ElasticNet** ($L_1 + L_2$).
4. **`RandomForest&XGBoost.ipynb`**:
   * Kết hợp các đặc trưng nghiệp vụ với mã hóa **Smoothed Target Encoding** cho biến nhiều nhãn (`Hãng Sản Xuất`, `Hãng Chip`).
   * Đánh giá mức độ đóng góp đặc trưng (Feature Importance), tối ưu siêu tham số bằng `RandomizedSearchCV` và kết hợp mô hình (**Blending Ensemble**).

---

## 4. Tổng kết & So sánh chi tiết kết quả các mô hình

> **Ghi chú về các tập dữ liệu:**
> * **Validation Set (286 mẫu):** Dùng để so sánh, kiểm chứng từng bước cải tiến Feature Engineering và dò tìm siêu tham số (Tuning).
> * **Test Set (287 mẫu):** Dữ liệu độc lập hoàn toàn, chỉ dùng để chấm điểm năng lực tổng quát hóa cuối cùng của các mô hình tốt nhất.

### 4.1. Mốc Tham Chiếu Ban Đầu (Baseline Models trên Validation Set)
*Được huấn luyện trên dữ liệu 21 cột gốc (chỉ qua tiền xử lý tối thiểu: điền Median/Mode + One-Hot Encoding cơ bản).*

| Mô hình Baseline | Val MAE (EUR) | Val RMSE (EUR) | Val $R^2$ Score | Đánh giá sơ bộ |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 58.20 € | 86.23 € | 0.6154 | Khó nắm bắt các tương tác phi tuyến phức tạp |
| **XGBoost (Mặc định)** | 51.46 € | 77.98 € | 0.6855 | Học tốt tương tác phi tuyến nhưng nhạy cảm tham số |
| **Random Forest (Mặc định)** | **52.32 €** | **75.88 €** | **0.7022** | Cân bằng và ổn định nhất khi chưa tinh chỉnh |

---

### 4.2. Nhật Ký Từng Bước Cải Tiến Mô Hình Tuyến Tính (`Linear_model.ipynb`)
*Đo đạc trên cùng một tập **Validation Set** qua từng vòng lặp cải tiến (kèm đối chiếu kết quả **Test Set** ở 2 cấu hình tốt nhất):*

| Vòng lặp cải tiến | Kỹ thuật áp dụng | Val MAE | Val RMSE | Val $R^2$ | Nhận xét chi tiết |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Bước 1** | Baseline OLS (21 cột gốc) | 58.20 € | 86.23 € | 0.6154 | Điểm khởi đầu tham chiếu |
| **Bước 2** | + 5 Đặc trưng Vật lý & Hiệu năng (`PPI`, `Pin/Dày`, `Hiệu năng`...) | 56.74 € | 83.92 € | 0.6357 | **Cải thiện rõ rệt:** Cung cấp thông tin phi tuyến giúp $R^2$ tăng +0.02 |
| **Bước 3** | Chỉ dùng Đặc trưng Thời gian (Định luật Moore riêng lẻ) | 59.57 € | 93.88 € | 0.5441 | Thiếu đặc trưng cấu hình chi tiết nên kết quả kém đi |
| **Bước 4** | Gộp cả Vật lý + Thời gian (chưa Scale/Log) | 58.00 € | 91.05 € | 0.5712 | Số chiều tăng vọt (98 cột) gây hiện tượng **đa cộng tuyến mạnh** |
| **Bước 5** | Gộp đầy đủ + Biến đổi Logarit (`y` & `ROM`) + `StandardScaler` | 56.48 € | 91.81 € | 0.5640 | Log-transform kéo giảm MAE, nhưng OLS thường vẫn chịu đa cộng tuyến |
| **Bước 6** | Thử ElasticNet ban đầu (`alpha=0.1, l1_ratio=0.5`) | 63.69 € | 98.87 € | 0.4944 | Lực phạt quá mạnh làm mô hình bị thiếu khớp (Underfitting) |
| **Bước 7** | **Tuning Lưới ElasticNet - Cấu hình Tối ưu theo MAE (`alpha=0.001, l1_ratio=0.3`)** | **52.23 €** (Test: **50.63 €**) | **80.91 €** (Test: **76.12 €**) | **0.6614** (Test: **0.6721**) | **Tốt nhất về MAE:** Phạt nhẹ vừa đủ để triệt tiêu đa cộng tuyến |
| **Bước 8** | **Tuning Lưới ElasticNet - Cấu hình Tối ưu theo RMSE (`alpha=0.001, l1_ratio=0.5`)** | **52.31 €** (Test: **51.12 €**) | **80.10 €** (Test: **77.65 €**) | **0.6681** (Test: **0.6588**) | **Tốt nhất về RMSE & $R^2$:** Tránh các cú dự đoán lệch lớn ở máy đắt tiền |

---

### 4.3. Quá Trình Cải Tiến & Tinh Chỉnh Mô Hình Cây (`RandomForest&XGBoost.ipynb`)
*Áp dụng bộ 34 đặc trưng (Feature Engineering + Smoothed Target Encoding $m=10$) và tinh chỉnh bằng `RandomizedSearchCV`:*

| Mô hình & Giai đoạn | Tập Train | Tập Validation | Nhận xét & Cấu hình |
| :--- | :---: | :---: | :--- |
| **Random Forest (Baseline FE - 34 biến)** | — | MAE: 50.62 € \| RMSE: 73.46 € \| $R^2$: 0.7208 | Vượt trội hơn hẳn Baseline chưa FE ($R^2$ tăng từ 0.7022 $\rightarrow$ 0.7208) |
| **XGBoost (Baseline FE - 34 biến)** | — | MAE: 48.26 € \| RMSE: 66.90 € \| $R^2$: 0.7685 | Cải thiện nhảy vọt nhờ Target Encoding và đặc trưng tương tác |
| **Random Forest (Sau lọc còn 28 biến)** | — | MAE: 50.31 € \| RMSE: 72.67 € \| $R^2$: 0.7268 | Lọc biến đóng góp < 0.5% giữ hiệu năng tương đương |
| **XGBoost (Sau lọc còn 28 biến)** | — | MAE: 48.57 € \| RMSE: 67.57 € \| $R^2$: 0.7639 | Giảm nhẹ so với giữ đủ 34 biến $\rightarrow$ quyết định giữ trọn 34 biến |
| **Random Forest (Đã Tuned - 34 biến)** | MAE: 22.47 € \| RMSE: 34.96 € \| $R^2$: 0.9348 | **MAE: 50.42 € \| RMSE: 72.86 € \| R^2: 0.7254** | `n_estimators=100, max_depth=15, min_samples_split=5` *(Test: MAE 48.40 € \| RMSE 69.42 € \| $R^2$ 0.7273)* |
| **XGBoost (Đã Tuned - 34 biến)** | MAE: 18.69 € \| RMSE: 25.10 € \| $R^2$: 0.9664 | **MAE: 46.19 € \| RMSE: 64.78 € \| R^2: 0.7829** | `n_estimators=300, max_depth=5, lr=0.05, subsample=0.9` *(Test: MAE 46.45 € \| RMSE 66.81 € \| $R^2$ 0.7474)* |

---

### 4.4. Điểm Số Đánh Giá Cuối Cùng Trên Tập Kiểm Thử (Test Set - 287 mẫu độc lập)

| Mô hình kiểm thử | Test MAE (EUR) | Test RMSE (EUR) | Test $R^2$ Score | Đánh giá tổng quát |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost (Đã Tuned)** | **46.45 €** | **66.81 €** | **0.7474** | **Mô hình đơn lẻ xuất sắc nhất toàn dự án ($R^2 \approx 74.7\%$)** |
| **Ensemble Blending (0.7 XGB + 0.3 RF)** | **46.52 €**  | **66.82 €**  | **0.7473** | **Độ ổn định cao, kết hợp ưu điểm giảm Bias và giảm Variance** |
| **Random Forest (Đã Tuned)** | 48.40 €  | 69.42 €  | 0.7273  | Cải thiện mạnh so với mô hình tuyến tính |
| **ElasticNet (Tối ưu theo MAE)** | 50.63 € | 76.12 € | 0.6721 | Tốt nhất trong nhóm mô hình Hồi quy Tuyến tính |
| **ElasticNet (Tối ưu theo RMSE)** | 51.12 € | 77.65 € | 0.6588 | Kiểm soát tốt đa cộng tuyến trên không gian 98 chiều |
---

## 5. Kết luận & Insight rút ra từ dự án

1. **Sự lột xác qua từng giai đoạn:**
   * Từ mức Baseline ban đầu (Linear Regression: $\text{MAE} = 58.20\text{ €}, R^2 = 0.6154$), việc áp dụng **Feature Engineering** + **Smoothed Target Encoding** + **Hyperparameter Tuning trên XGBoost** đã kéo sai số trên tập Test xuống chỉ còn **46.45 €** và nâng hệ số xác định $R^2$ lên tới **0.7474** (giải thích gần **75%** biến thiên giá điện thoại).
2. **Sức mạnh của Điều chuẩn & Mô hình Cây:**
   * Ở nhóm tuyến tính, **ElasticNet** (`alpha = 0.001`) giúp giải quyết triệt để đa cộng tuyến của 98 cột, đưa Test MAE về **50.63 €** ($R^2 = 0.6721$).
   * Ở nhóm phi tuyến, **XGBoost** và **Random Forest** khai thác cực tốt các biến tỷ lệ theo thời gian (Định luật Moore) và giá trị thương hiệu (`Hang_San_Xuat_TE`), tạo ra bước nhảy vọt về độ chính xác.
3. **Top đặc trưng chi phối giá trị smartphone:**
   * `ROM (GB)` (Dung lượng bộ nhớ trong), `Cam_So_Voi_Nam` (Độ phân giải camera so với thời điểm ra mắt), `PPI` (Mật độ điểm ảnh màn hình), `Hang_San_Xuat_TE` (Thương hiệu sản phẩm) và `Chi_So_Hieu_Nang` (RAM $\times$ Tổng xung nhịp CPU).
