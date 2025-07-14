# Trích xuất thông tin trên thẻ sinh viên trường Đại học Khoa học - Đại học Huế

## 1. Giới thiệu

Đây là một dự án nghiên cứu khoa học với mục tiêu **tự động trích xuất thông tin từ ảnh thẻ sinh viên** bằng cách kết hợp các kỹ thuật **xử lý ảnh** và **nhận dạng ký tự quang học (OCR)**. Ứng dụng này có thể hỗ trợ số hóa thông tin sinh viên, tự động nhập dữ liệu vào hệ thống quản lý và nâng cao hiệu quả hành chính trong các trường đại học.

## Mục tiêu

- Phát hiện vùng chứa thông tin trên thẻ sinh viên (họ và tên, mã số sinh viên, ngày sinh, lớp, khóa học).
- Tiền xử lý ảnh để tối ưu kết quả nhận dạng.
- Áp dụng mô hình OCR để trích xuất văn bản từ vùng ảnh đã phát hiện.
- Chuẩn hóa và hiển thị thông tin dưới dạng có cấu trúc.

## Công nghệ sử dụng

- **OpenCV** – Xử lý ảnh và phát hiện vùng quan tâm.
- **PaddleOCR** – Nhận dạng ký tự quang học.
- **PaddleLite** – Chạy mô hình AI nhẹ, có thể deploy trên thiết bị di động.
- **LabelMe** – Gán nhãn dữ liệu (nếu huấn luyện thêm).
- **Flutter** – Giao diện người dùng trên mobile.