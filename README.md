# CNN Satellite Classification using ResNet-34

[![Python](https://img.shields.io/badge/Python-3.10.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Data-Aqua%20MODIS-green.svg)](https://modis.gsfc.nasa.gov/data/)

## 📝 Giới thiệu
Đồ án cuối kỳ môn **Trí tuệ nhân tạo (AI)**. Dự án thực hiện phân loại ảnh vệ tinh từ cảm biến **MODIS** của vệ tinh **Aqua** (NASA). Mục tiêu là nhận diện và phân loại các kiểu che phủ bề mặt (ví dụ: mây, nước, rừng, băng tuyết,...) dựa trên kiến trúc mạng Deep Learning **ResNet-34**.

## 🛰️ Dữ liệu (Dataset)
Dữ liệu được trích xuất từ vệ tinh Aqua thông qua bộ cảm biến MODIS:
- **Nguồn:** NASA Earthdata / MODIS Aqua.
- **Tiền xử lý:**
  - Resize ảnh về kích thước chuẩn $224 \times 224$ cho ResNet.
  - Chuyển đổi định dạng ảnh và chuẩn hóa kênh màu (Normalization).
  - Tăng cường dữ liệu (Augmentation): Random Crop, Horizontal Flip để tăng độ tổng quát cho mô hình.

## 🏗️ Kiến trúc mô hình (Model Architecture)
Sử dụng mô hình **ResNet-34** (Residual Network 34 layers) với phương pháp **Transfer Learning**:
- **Residual Connections:** Giúp giải quyết vấn đề biến mất đạo hàm (vanishing gradient) trong mạng sâu.
- **Pre-trained:** Tận dụng trọng số từ tập ImageNet để tăng tốc độ hội tụ.
- **Custom Head:** Thay đổi lớp Fully Connected cuối cùng để phù hợp với số lượng nhãn phân loại của đồ án.



## 🛠️ Cài đặt & Sử dụng

### 1. Cài đặt môi trường
```bash
git clone [[https://github.com/thinha1/CNN_SatelliteClassification.git](https://github.com/Thinha1/LocalRainfallClassifcation.git)
cd CNN_SatelliteClassification
pip install -r requirement.txt
