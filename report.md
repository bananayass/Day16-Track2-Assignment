Mô hình LightGBM được huấn luyện trên CPU EC2 t3.medium với bộ dữ liệu Credit Card Fraud Detection gồm 284,807 giao dịch.
Thời gian tải dữ liệu là khoảng 2.34 giây và thời gian huấn luyện là khoảng 11.26 giây.
Mô hình đạt AUC-ROC 0.9428, cho thấy khả năng phân biệt giao dịch gian lận và bình thường khá tốt.
Accuracy đạt 99.949%, F1-Score đạt 0.8497.
Precision đạt 0.8632 và Recall đạt 0.8367.
Inference latency cho một giao dịch chỉ khoảng 1.29 ms.
Hệ thống đạt throughput khoảng 65,516 giao dịch/giây trong phép đo benchmark.
Kết quả cho thấy LightGBM có thể huấn luyện và inference nhanh trên CPU t3.medium đối với workload của bài lab.
Thời gian chạy và tốc độ xử lý có thể thay đổi nhẹ giữa các lần chạy do điều kiện thực tế của EC2 tại từng thời điểm.
# Trần Nguyễn Trí Dũng - 2A202602784