<!-- # Life Expectancy Prediction - WHO Dataset

Dự đoán tuổi thọ trung bình các quốc gia dựa trên yếu tố y tế - xã hội.

## Cấu trúc
- `data/`: Dataset gốc
- `src/source.ipynb`: Code chính
- `results/figures/`: Biểu đồ
- `dimensionality-reduced-data/`: Dữ liệu giảm chiều

## Chạy
1. `pip install -r requirements.txt`
2. Mở `src/source.ipynb` → Chạy từng cell -->
# Dự Đoán Tuổi Thọ Trung Bình (Life Expectancy Prediction)

## Tổng quan
Dự án được thực hiện trong khuôn khổ bài tập lớn môn Học Máy (MAT3533) - HUS, nhằm xây dựng mô hình dự đoán tuổi thọ trung bình các quốc gia dựa trên các yếu tố y tế - xã hội từ bộ dữ liệu WHO Life Expectancy. Các bước chính bao gồm:

- **Tiền xử lý dữ liệu**: Xử lý missing values bằng Mean Imputation (2563 giá trị - 3.97%), chuẩn hóa dữ liệu (StandardScaler), mã hóa biến phân loại Status (Developed/Developing), và loại bỏ biến Country để tránh curse of dimensionality.
- **Giảm chiều**: 
  - PCA (giữ 95% phương sai): Giảm từ 20 chiều xuống 15 chiều
  - SelectKBest (f_regression): Chọn 10 features quan trọng nhất dựa trên F-statistic
- **Mô hình hồi quy**:
  - **K-Nearest Neighbors (KNN)**: Sử dụng khoảng cách Manhattan, k = √n
  - **Linear Regression**: Ordinary Least Squares (OLS)
- **Đánh giá**: So sánh hiệu suất trên dữ liệu gốc, PCA, và SelectKBest với 3 tỉ lệ chia (80:20, 70:30, 60:40) dựa trên R², MSE, RMSE, và phân tích residuals.
- **Kết quả trực quan**: Biểu đồ phân tán PCA, correlation matrix, residual plots, và so sánh R² theo tỉ lệ chia.

Dự án được tổ chức để dễ dàng tái hiện, với mã nguồn, tài liệu, và hướng dẫn chi tiết.

### Dữ liệu: WHO Life Expectancy Dataset
Bộ dữ liệu Life Expectancy từ WHO bao gồm 2938 mẫu từ 193 quốc gia (2000-2015), mỗi mẫu được mô tả qua 22 thuộc tính bao gồm các yếu tố y tế (Adult Mortality, HIV/AIDS, BMI, vaccination rates), kinh tế (GDP, expenditure), và xã hội (Schooling, Income composition). Mục tiêu là dự đoán tuổi thọ trung bình (Life expectancy) - một biến liên tục (36.3 - 89.0 năm).

Thông tin chi tiết về dataset có thể tìm thấy ở [WHO Life Expectancy Data](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)

## Thông tin thêm
**Giảng viên hướng dẫn:** [TS. Cao Văn Chung](https://hus.vnu.edu.vn/gioi-thieu/can-bo/danh-sach-can-bo/cao-van-chung-3004.html)

**Sinh viên thực hiện:**
- Đinh Quang Tiến - 23001943
- Nguyễn Đức Quang - 23001918
- Hà Minh Quang - 23001916

**Phân chia công việc:**
- Tiền xử lý dữ liệu: Xử lý missing values, encoding, chuẩn hóa
- Giảm chiều: PCA, SelectKBest, phân tích feature importance
- Mô hình hồi quy: KNN (3 tỉ lệ chia × 3 loại dữ liệu), Linear Regression
- Đánh giá và trực quan hóa: So sánh R², MSE, residual analysis
- Báo cáo và tài liệu: Viết báo cáo chi tiết về phương pháp và kết quả

## Cấu trúc thư mục

```plain
life-expectancy-prediction/
├── src/                           # Mã nguồn
│   └── source.ipynb               # Notebook chính (phân tích đầy đủ)
├── data/                          # Thư mục dữ liệu
│   └── Life Expectancy Data.csv   # Dữ liệu gốc WHO
├── dimensionality-reduced-data/   # Dữ liệu đã giảm chiều
│   ├── X_train_pca.npy            # Train set sau PCA
│   ├── X_test_pca.npy             # Test set sau PCA
│   ├── X_train_kbest.npy          # Train set sau SelectKBest
│   ├── X_test_kbest.npy           # Test set sau SelectKBest
│   ├── y_train.npy                # Target train
│   ├── y_test.npy                 # Target test
│   └── selected_features.txt      # Danh sách features được chọn
├── results/                       # Kết quả thực nghiệm
│   ├── figures/                   # Các biểu đồ/hình ảnh
│   │   ├── target_distribution.png
│   │   ├── missing_values_analysis.png
│   │   ├── pca_explained_variance.png
│   │   ├── pca_scatter_plot.png
│   │   ├── selectkbest_feature_importance.png
│   │   ├── selectkbest_correlation_matrix.png
│   │   └── pca_features_correlation.png
│   ├── knn_result/                # Kết quả KNN
│   │   ├── K-nearest neighbors_comparison.csv
│   │   ├── r2_by_split_ratio.png
│   │   └── delta_r2_comparison.png
│   └── linear_result/             # Kết quả Linear Regression
│       ├── linear_regression_comparison.csv
│       ├── r2_by_split_ratio.png
│       └── delta_r2_comparison.png
├── requirements.txt               # Danh sách thư viện cần thiết
├── README.md                      # Hướng dẫn tổng quan
└── .gitignore                     # Tệp bỏ qua Git
```

## Thiết lập

1. Clone repository về máy (chọn đường dẫn phù hợp):
   ```bash
   git clone https://github.com/ducqwang/life-expectancy-prediction.git
   cd life-expectancy-prediction
   ```

2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

   **Các thư viện chính:**
   - numpy >= 1.21.0
   - pandas >= 1.3.0
   - scikit-learn >= 1.0.0
   - matplotlib >= 3.4.0
   - seaborn >= 0.11.0

## Để chạy chương trình

Tất cả source code của dự án được trình bày đầy đủ và giải thích chi tiết trong `src/source.ipynb`. Notebook được chia thành các phần:

### Phần 1: Tiền xử lý dữ liệu
- Đọc và khám phá dữ liệu
- Xử lý missing values (Mean Imputation)
- Mã hóa biến phân loại (Status)
- Chuẩn hóa dữ liệu (StandardScaler)

### Phần 2: Phân tích và giảm chiều
- **2.1**: Tách train/test (70:30)
- **2.2**: PCA - Giảm xuống 15 components (95% variance)
- **2.3**: SelectKBest - Chọn 10 features tốt nhất (f_regression)
- **2.4**: So sánh và trực quan hóa
- **2.5**: Lưu dữ liệu đã giảm chiều

### Phần 3: Mô hình hồi quy
- **3.1**: K-Nearest Neighbors (KNN)
  - 3 tỉ lệ chia: 80:20, 70:30, 60:40
  - 3 loại dữ liệu: Gốc, PCA, SelectKBest
  - Đánh giá R², MSE, RMSE, residual analysis
- **3.2**: Linear Regression
  - Tương tự KNN với 9 thí nghiệm
  - So sánh hiệu suất và overfit (ΔR²)

**Chạy từng cell theo thứ tự trong notebook để tái hiện kết quả.**

## Kết quả chính

### Tiền xử lý
- Xử lý thành công 2563 missing values (3.97%)
- Chuẩn hóa: mean ≈ 0, std ≈ 1

### Giảm chiều
- **PCA**: 20 → 15 chiều (giảm 25%), giữ 95% variance trên train và 96.55% trên test
- **SelectKBest**: Top 10 features quan trọng nhất
  1. Schooling (F-score: 2186.35)
  2. Income composition of resources (1953.29)
  3. Adult Mortality (1936.17)
  4. HIV/AIDS (935.39)
  5. BMI (926.69)

### Mô hình hồi quy

#### KNN (Kết quả tốt nhất - SelectKBest, tỉ lệ 60:40)
- Test R²: **0.8860**
- Test MSE: 10.4247
- RMSE: 3.2287
- ΔR² (overfit): 0.0116 (Không overfit)

#### Linear Regression (Kết quả tốt nhất - Dữ liệu gốc, tỉ lệ 70:30)
- Test R²: **0.8126**
- Test MSE: 17.5262
- ΔR² (overfit): 0.0095 (Không overfit)

**Nhận xét:**
- SelectKBest cho kết quả tốt nhất với KNN (giảm ΔR² 33.3% so với dữ liệu gốc)
- Linear Regression ổn định hơn với dữ liệu gốc
- Tất cả mô hình không có dấu hiệu overfit (ΔR² < 0.05)

## Kết quả trực quan

Các biểu đồ và hình ảnh được lưu trong:
- `results/figures/`: Phân tích dữ liệu, PCA, SelectKBest
- `results/knn_result/`: So sánh KNN
- `results/linear_result/`: So sánh Linear Regression

## Tài liệu tham khảo

- [WHO Life Expectancy Dataset](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PCA Tutorial](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)


---
**Lưu ý:** Đây là dự án học tập, kết quả chỉ mang tính chất tham khảo và không nên sử dụng cho mục đích y tế thực tế.