# Phát hiện trạng thái thị trường bằng Gaussian HMM

Dự án sử dụng **Mô hình Markov ẩn Gaussian** (*Gaussian Hidden Markov Model – Gaussian HMM*) để phát hiện và đánh giá các trạng thái ẩn của thị trường tài chính từ dữ liệu giá cổ phiếu, đối chiếu với baseline **Gaussian Mixture Model (GMM)** để kiểm tra mức độ cần thiết của việc mô hình hóa phụ thuộc thời gian giữa các trạng thái.

> Dự án này được tách ra từ một đề tài nhóm 3 người (Gaussian HMM, MS-AR, GMM/Time-Series K-means). Do yêu cầu thay đổi, mỗi thành viên làm độc lập; repo này chỉ giữ lại phần thuật toán chính (Gaussian HMM) và một baseline so sánh (GMM), dùng cùng bộ dữ liệu và cùng cách chia walk-forward như bản gốc.

## Thông tin đề tài

| Hạng mục | Nội dung |
| --- | --- |
| Tên đề tài | Phát hiện và đánh giá các trạng thái ẩn của thị trường tài chính bằng mô hình Markov ẩn Gaussian |
| Bộ dữ liệu | Stock Market Dataset for Predictive Analysis – Kaggle |
| Code tham khảo | [hidden-regime/hidden-regime](https://github.com/hidden-regime/hidden-regime), [hmmlearn/hmmlearn](https://github.com/hmmlearn/hmmlearn) |
| Thuật toán chính | Gaussian Hidden Markov Model (Gaussian HMM) |
| Baseline so sánh | Gaussian Mixture Model (GMM) |

## Mô tả bài toán

Thị trường có thể ở nhiều chế độ như tăng trưởng, suy giảm, đi ngang hoặc biến động mạnh. Những chế độ này không được ghi nhận trực tiếp và không có nhãn sẵn; dữ liệu chỉ chứa các đại lượng quan sát như giá, khối lượng giao dịch, chỉ báo kỹ thuật và điểm tâm lý thị trường.

Bài toán thuộc nhóm **học máy không giám sát trên dữ liệu chuỗi thời gian**. Gaussian HMM được dùng để suy luận trạng thái thị trường ở từng thời điểm, đồng thời mô hình hóa cách thị trường chuyển đổi giữa các trạng thái đó theo thời gian. GMM đóng vai trò baseline: nó phân cụm từng quan sát độc lập, không có khái niệm chuyển trạng thái theo thời gian. So sánh hai mô hình giúp trả lời câu hỏi: **liệu việc mô hình hóa chuỗi Markov ẩn có thực sự cần thiết, hay phân cụm tĩnh đã đủ tốt?**

## Mô hình

Quan sát tại thời điểm \(t\), ký hiệu \(X_t\), được sinh ra từ trạng thái ẩn \(Z_t\).

- **Gaussian HMM**: chuỗi trạng thái \(Z_t\) tuân theo chuỗi Markov bậc 1, quan sát mỗi trạng thái là Gaussian đa biến: `P(Zₜ | Zₜ₋₁)`, với `Xₜ | Zₜ = k ~ N(μₖ, Σₖ)`. Học bằng **Baum–Welch**, suy luận bằng **Viterbi** (chuỗi trạng thái khả dĩ nhất) và **Forward–Backward** (xác suất hậu nghiệm).
- **GMM (baseline)**: mỗi \(X_t\) được gán vào một trong K phân phối Gaussian độc lập theo thời gian — không có \(P(Z_t \mid Z_{t-1})\) học được. Ma trận chuyển trạng thái chỉ có thể tính **thực nghiệm** (empirical) từ chuỗi nhãn dự đoán, để đối chiếu với ma trận Gaussian HMM học được.

Với xác suất tự chuyển `pᵢᵢ`, thời gian lưu trú kỳ vọng là:

```text
E[Dᵢ] = 1 / (1 − pᵢᵢ)
```

## Dữ liệu và đặc trưng đầu vào

Bộ dữ liệu **Stock Market Dataset for Predictive Analysis** (Kaggle). File đã tiền xử lý: `data/processed/market_features.csv`, sinh bởi `src/shared/preprocess_data.py`. Chi tiết ở [data/processed/README.md](data/processed/README.md).

Bộ đặc trưng chuẩn hóa (`_z`) dùng chung cho cả Gaussian HMM và GMM:

- lợi suất log (`log_return_z`), lợi suất đơn (`simple_return_z`), lợi suất trong phiên (`intraday_return_z`)
- biên độ dao động trong phiên (`intraday_range_z`), thay đổi khối lượng (`volume_log_change_z`, `volume_zscore_20_z`)
- độ biến động cửa sổ 5/20 phiên (`volatility_5_z`, `volatility_20_z`), trung bình lợi suất 5 phiên (`return_mean_5_z`)
- chỉ báo kỹ thuật RSI, MACD, điểm tâm lý thị trường (`RSI_z`, `MACD_z`, `Sentiment_z`)

Dữ liệu được chia theo thời gian thành `train` / `validation` / `test`, cùng các fold `walk_forward_fold` để đánh giá ngoài mẫu. Chuẩn hóa dùng mean/std tính từ tập `train` để tránh rò rỉ dữ liệu.

## Cấu trúc dự án

```text
.
├── data/
│   ├── raw/                          # Dữ liệu gốc từ Kaggle
│   └── processed/                    # Đặc trưng đã tiền xử lý (dùng chung cho cả 2 mô hình)
├── configs/
│   ├── hmm.yaml                      # Cấu hình thí nghiệm Gaussian HMM
│   └── gmm.yaml                      # Cấu hình thí nghiệm GMM (baseline)
├── src/
│   ├── shared/                       # Tiện ích dùng chung: I/O, walk-forward split, metrics
│   ├── gaussian_hmm/                 # Thuật toán chính
│   │   ├── train.py                  # Baum–Welch, chọn số trạng thái bằng AIC/BIC
│   │   ├── inference.py              # Viterbi, Forward–Backward
│   │   ├── evaluate.py               # Walk-forward, ma trận chuyển, thống kê trạng thái
│   │   └── visualize.py              # Biểu đồ trạng thái, chuyển tiếp, AIC/BIC
│   └── gmm/                          # Baseline
│       ├── train.py                  # Fit GaussianMixture, chọn số cụm bằng AIC/BIC
│       ├── evaluate.py               # Suy luận + walk-forward + ma trận chuyển thực nghiệm
│       └── visualize.py              # Biểu đồ cụm, AIC/BIC
├── notebooks/
│   └── 01_gaussian_hmm.ipynb
├── models/
│   ├── gaussian_hmm/                 # best_hmm.pkl
│   └── gmm/                          # best_gmm.pkl (sinh ra khi chạy train.py)
├── reports/
│   ├── figures/{gaussian_hmm,gmm}/
│   └── tables/{gaussian_hmm,gmm}/
├── tests/
│   ├── gaussian_hmm/
│   └── gmm/
└── main.py
```

## Cách chạy

Cài đặt phụ thuộc (khuyến nghị dùng [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
# hoặc: pip install -r requirements.txt
```

Dữ liệu đã tiền xử lý sẵn có trong `data/processed/`. Chỉ cần chạy lại tiền xử lý nếu muốn tái tạo từ đầu:

```bash
python -m src.shared.preprocess_data
```

Pipeline Gaussian HMM (thuật toán chính):

```bash
python -m src.gaussian_hmm.train      # huấn luyện + chọn mô hình tốt nhất
python -m src.gaussian_hmm.inference  # Viterbi + Forward-Backward
python -m src.gaussian_hmm.evaluate   # ma trận chuyển, thống kê trạng thái, walk-forward
python -m src.gaussian_hmm.visualize  # biểu đồ
```

Pipeline GMM (baseline so sánh):

```bash
python -m src.gmm.train
python -m src.gmm.evaluate    # suy luận + ma trận chuyển thực nghiệm + walk-forward
python -m src.gmm.visualize
```

Chạy test:

```bash
python -m unittest discover tests
```

Model, bảng kết quả và biểu đồ của Gaussian HMM trong `models/gaussian_hmm/` và `reports/*/gaussian_hmm/` là kết quả đã chạy sẵn (giữ lại làm tham khảo); baseline GMM cần chạy pipeline ở trên để sinh ra tương ứng trong `models/gmm/` và `reports/*/gmm/`.

## Đầu ra mong đợi

- Chuỗi trạng thái/cụm thị trường theo thời gian, xác suất hậu nghiệm tại mỗi thời điểm.
- Ma trận chuyển trạng thái (học được với Gaussian HMM, thực nghiệm với cả hai mô hình).
- Thời gian lưu trú kỳ vọng của từng trạng thái.
- Thống kê mô tả (lợi suất, biến động, khối lượng) theo trạng thái/cụm.
- Kết quả walk-forward validation và bảng so sánh Gaussian HMM với GMM.

## Câu hỏi nghiên cứu

- Thị trường có thể được phân chia thành bao nhiêu trạng thái hợp lý?
- Mỗi trạng thái khác nhau như thế nào về lợi suất và mức độ rủi ro?
- Trạng thái nào có xu hướng duy trì lâu nhất, và điều đó có ý nghĩa gì về mặt tài chính?
- Việc mô hình hóa chuyển trạng thái theo thời gian (Gaussian HMM) có cải thiện đáng kể so với phân cụm tĩnh (GMM) không, xét trên log-likelihood ngoài mẫu và tính ổn định của trạng thái?

## Công nghệ

- Python, `pandas`, `numpy`, `scikit-learn`, `hmmlearn`, `pyyaml`
- `matplotlib`, `seaborn` để trực quan hóa
- Jupyter Notebook

## Lưu ý

Kết quả chỉ phục vụ mục đích nghiên cứu và học thuật, không phải khuyến nghị đầu tư. Thị trường tài chính có rủi ro; trạng thái suy luận từ dữ liệu quá khứ không bảo đảm dự báo chính xác trong tương lai.
