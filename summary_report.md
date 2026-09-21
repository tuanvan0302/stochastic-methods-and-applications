# Báo cáo tóm tắt: Nhận diện trạng thái ẩn của thị trường bằng Gaussian HMM

## Ý tưởng

**Bối cảnh**: Trong kinh tế thị trường, ta chỉ biết được mỗi ngày giao dịch: giá bán ra, giá mua vào, giá lớn nhất, giá nhỏ nhất, số lượng, .... Tiềm ẩn trong những cái giao dịch này là một trạng thái ẩn của thị trường. Mục đích ta cần phải mô hình hóa được những cái trạng thái ẩn này, tìm được xác suất chuyển từ trạng thái này sang trạng thái kia như nào và dự báo được với dữ liệu hiện tại thì trong tương lai gần nó sẽ là trạng thái gì theo xác suất.

Bài toán gồm 3 nhiệm vụ chính:

1.  Khám phá xem thị trường có thể được mô tả thành bao nhiêu trạng thái ẩn (hidden states).
2.  Mô hình hóa bài toán từ dữ liệu:
    *   Học đặc trưng thống kê của từng trạng thái.
    *   Tìm xác suất chuyển trạng thái giữa các hidden states (transition matrix).
3.  Suy luận với một mẫu dữ liệu mới: xác định trạng thái hiện tại là gì, rồi dựa vào transition matrix và đặc trưng của từng trạng thái để dự báo xác suất chuyển sang các trạng thái khác trong tương lai gần.

---

### 1. Bài toán đang đề cập là gì?

*   **Bối cảnh & Vấn đề**: Thị trường tài chính là một hệ thống phi dừng (non-stationary); các đặc tính thống kê như lợi suất và độ biến động không cố định mà thay đổi theo thời gian qua các giai đoạn tăng trưởng, đi ngang, suy thoái hoặc bùng nổ biến động. Các giả định cổ điển coi phân phối lợi suất là tĩnh dễ dẫn đến sai lệch mô hình nghiêm trọng.
*   **Bản chất bài toán**: Đây là bài toán **học không giám sát (unsupervised learning) trên dữ liệu chuỗi thời gian tài chính**. Mục tiêu là suy luận biến trạng thái rời rạc ẩn $Z_t \in \{1, \dots, K\}$ tại mỗi thời điểm $t$ từ vector quan sát đa chiều $X_t \in \mathbb{R}^{12}$ (gồm 12 đặc trưng tài chính: lợi suất, độ biến động, khối lượng, các chỉ báo kỹ thuật RSI, MACD và điểm tâm lý thị trường).
*   **Ý nghĩa thực tiễn**: Nhận diện tự động các trạng thái ẩn để phục vụ quản trị rủi ro (tái cân bằng danh mục khi biến động cao), xây dựng chiến lược giao dịch thích ứng (chuyển đổi giữa trend-following và mean-reversion) và phân tích hành vi thị trường trong các giai đoạn bất thường.

---

### 2. Các phương pháp sử dụng và đối tượng so sánh

*   **Phương pháp nghiên cứu chính (đề xuất)**: **Mô hình Markov ẩn Gaussian (Gaussian Hidden Markov Model - Gaussian HMM)**.
*   **Phương pháp so sánh (Baseline)**: **Mô hình phân cụm tĩnh Gaussian Mixture Model (GMM)**.
*   **Mục đích đối chiếu**: So sánh giữa hai hướng tiếp cận:
    1.  *Mô hình chuỗi có nhớ (Sequential / state-space)* như HMM: Có mô hình hóa tính phụ thuộc thời gian và ma trận xác suất chuyển trạng thái giữa các chế độ.
    2.  *Phân cụm tĩnh (Static clustering)* như GMM: Giả định các quan sát độc lập và cùng phân phối (i.i.d), hoàn toàn bỏ qua thứ tự thời gian.

---

### 3. Ý tưởng chi tiết của từng phương pháp

#### 3.1. Ý tưởng của Gaussian Hidden Markov Model (Gaussian HMM)

Gaussian HMM là một dạng mạng Bayes động (dynamic Bayesian network) kết hợp song hành hai chuỗi: chuỗi trạng thái ẩn $Z_t$ không thể quan sát trực tiếp và chuỗi dữ liệu thực tế $X_t$ được phát xạ từ các trạng thái đó.

*   **Hai giả thiết cốt lõi**:
    1.  *Giả thiết Markov bậc nhất*: Trạng thái tại thời điểm $t$ ($Z_t$) chỉ phụ thuộc trực tiếp vào trạng thái ngay trước đó ($Z_{t-1}$): $P(Z_t \mid Z_1, \dots, Z_{t-1}) = P(Z_t \mid Z_{t-1})$. Tính chất này thể hiện "quán tính" của thị trường thông qua ma trận xác suất chuyển trạng thái $A = [A_{ij}]_{K \times K}$, trong đó $A_{ij} = P(Z_t = j \mid Z_{t-1} = i)$.
    2.  *Giả thiết phát xạ Gaussian*: Tại mỗi thời điểm $t$, quan sát $X_t$ tuân theo phân phối chuẩn đa biến tương ứng với trạng thái $Z_t = k$: $X_t \mid (Z_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$, với vector kỳ vọng $\mu_k$ và ma trận hiệp phương sai $\Sigma_k$.
*   **Bộ tham số mô hình**: Ký hiệu $\theta = (\pi, A, B)$, gồm vector phân phối ban đầu $\pi$, ma trận chuyển $A$, và tập tham số phát xạ $B = \{\mu_k, \Sigma_k\}_{k=1}^K$.
*   **Cơ chế học tham số (Thuật toán Baum–Welch / EM)**:
    *   Do chuỗi trạng thái $Z_t$ bị ẩn, hàm hợp lý không thể tối ưu hóa trực tiếp dạng đóng. Thuật toán Baum–Welch luân phiên hai bước:
        *   **E-step (Kỳ vọng)**: Sử dụng thuật toán Forward–Backward để tính biến xuôi $\alpha_t(i)$ và biến ngược $\beta_t(i)$, từ đó suy ra xác suất hậu nghiệm $\gamma_t(i) = P(Z_t = i \mid X_{1:T}, \theta)$ và xác suất chuyển đồng thời $\xi_t(i, j) = P(Z_t = i, Z_{t+1} = j \mid X_{1:T}, \theta)$.
        *   **M-step (Cực đại hóa)**: Cập nhật lại bộ tham số $\pi_i, A_{ij}, \mu_k, \Sigma_k$ nhằm tối đa hóa kỳ vọng log-likelihood.
*   **Cơ chế suy luận chuỗi trạng thái tối ưu (Thuật toán Viterbi)**:
    *   Để tránh mâu thuẫn gán nhãn giữa các thời điểm liền kề, quy hoạch động Viterbi tìm toàn cục chuỗi trạng thái $\hat{Z}_{1:T}$ có xác suất kết hợp cao nhất trên toàn bộ khoảng thời gian.
*   **Thời gian duy trì kỳ vọng của trạng thái**:
    *   Thời gian liên tục $D_i$ mà chuỗi ở lại trạng thái $i$ tuân theo phân phối hình học; số phiên giao dịch kỳ vọng là $E[D_i] = \frac{1}{1 - A_{ii}}$.

#### 3.2. Ý tưởng của Gaussian Mixture Model (GMM - Baseline)

GMM là mô hình phân cụm mật độ tĩnh, coi phân phối của dữ liệu quan sát là hỗn hợp của $K$ thành phần Gaussian.

*   **Giả thiết cốt lõi**: Giả định các quan sát $X_t$ là **độc lập và cùng phân phối (i.i.d)**, hoàn toàn không xét đến thứ tự hay khoảng cách thời gian giữa các phiên.
*   **Cấu trúc mật độ**: $p(X_t) = \sum_{k=1}^K w_k \mathcal{N}(X_t \mid \mu_k, \Sigma_k)$, trong đó $w_k$ là trọng số hỗn hợp cố định áp dụng cho mọi thời điểm $t$, không phụ thuộc vào trạng thái trước đó ($P(Z_t = k \mid Z_{t-1}) = P(Z_t = k) = w_k$).
*   **Cơ chế học & Gán nhãn**:
    *   Thuật toán EM cập nhật trách nhiệm (responsibility) $r_{tk} = P(Z_t = k \mid X_t)$ dựa thuần túy vào khoảng cách mật độ của điểm $X_t$ hiện tại tới các tâm cụm mà bỏ qua hoàn toàn lịch sử trước đó.
*   **Ma trận chuyển thực nghiệm (Hậu kỳ)**:
    *   GMM không học ma trận chuyển trạng thái trong quá trình tối ưu. Ma trận chuyển $\hat{A}^{GMM}$ chỉ được tính hậu kỳ dựa trên tần suất chuyển dịch quan sát được sau khi đã gán nhãn tĩnh cứng cho từng phiên: $\hat{A}^{GMM}_{ij} = \frac{\sum I(\hat{Z}_t = i, \hat{Z}_{t+1} = j)}{\sum I(\hat{Z}_t = i)}$.

#### 3.3. So sánh sự khác biệt then chốt về mặt ý tưởng

| Tiêu chí | Gaussian HMM | GMM (Baseline) |
| :--- | :--- | :--- |
| **Mô hình hóa thời gian** | Tường minh qua xích Markov $P(Z_t \mid Z_{t-1})$. | Bỏ qua hoàn toàn, giả định i.i.d. |
| **Ma trận chuyển $A$** | Được tối ưu hóa trực tiếp trong quá trình học (tốn thêm $K(K-1)$ tham số). | Không học trong mô hình; chỉ tính thực nghiệm hậu kỳ. |
| **Gán nhãn trạng thái** | Dùng Viterbi giải mã chuỗi tối ưu toàn cục. | Gán độc lập tại từng thời điểm theo trách nhiệm $r_{tk}$. |

---

### 4. Kết quả thực nghiệm

*   **Lựa chọn mô hình tối ưu (theo BIC & ràng buộc tỷ trọng trạng thái $\ge 1\%$)**:
    *   **Gaussian HMM**: Chọn **4 trạng thái, ma trận hiệp phương sai đầy đủ (4-full)**. Số tham số tự do $p = 375$, BIC train = $-48.051,6$.
    *   **GMM**: Chọn **5 cụm, ma trận hiệp phương sai đầy đủ (5-full)**. Số tham số tự do $p = 454$, BIC train = $-33.446,5$. GMM cần nhiều cụm hơn HMM để bù đắp việc thiếu cấu trúc thời gian.
*   **Khả năng tổng quát hóa trên tập Validation**:
    *   Log-likelihood trung bình/mẫu trên tập validation: Gaussian HMM đạt **$-0,887$**, cao hơn hẳn so với GMM đạt **$-1,298$**.
*   **Đặc trưng các trạng thái**:
    *   Cả hai mô hình đều tách ra được một trạng thái **"Biến động mạnh"** rõ rệt (độ lệch chuẩn lợi suất 40–55%, chiếm 5–7% dữ liệu) và các trạng thái **"Đi ngang / hỗn hợp"** chiếm đa số.
*   **Tính bền vững & Thời gian duy trì kỳ vọng ($E[D_i]$)**:
    *   Gaussian HMM có xác suất tự-chuyển $A_{ii}$ rất cao ($0,874 - 0,991$), thời gian duy trì kỳ vọng dài (từ **8 đến 116 phiên**). Ma trận chuyển học được trùng khớp gần như tuyệt đối với ma trận thực nghiệm.
    *   GMM có xác suất tự-chuyển thấp hơn ($0,752 - 0,938$), trạng thái đổi liên tục với thời gian duy trì rất ngắn (chỉ **4 đến 16 phiên**).
*   **Đánh giá ngoài mẫu Walk-Forward (5 Folds)**:
    *   Gaussian HMM đạt log-likelihood cao hơn GMM ở **3/5 fold** (Fold 2, 3, 4).
    *   Tuy nhiên ở Fold 5, HMM bị phạt nặng hơn do quá "tự tin" vào cấu trúc thời gian đã học khi dữ liệu ngoài mẫu biến đổi phân phối đột ngột.
*   **Kết luận chung**: Việc tích hợp cấu trúc xích Markov (HMM) mang lại ưu thế rõ rệt so với phân cụm tĩnh (GMM) về tiêu chí BIC, khả năng khớp dữ liệu validation và tạo ra cấu trúc trạng thái ổn định, bền vững có ý nghĩa kinh tế.

---