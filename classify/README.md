# Phân loại thẻ sinh viên

## 1. Chuẩn bị tập dữ liệu
- Tập dữ liệu sẽ được chia thành các thư mục con theo từng loại thẻ.
- Cấu trúc thư mục tập dữ liệu đã được gán nhãn:
    ```
    datasets/
    ├── classify/ 
    │   ├── library/
    │   │   ├── library_card_1.jpg
    │   │   ├── library_card_2.jpg
    │   │   └── ...
    │   └── student/
    │       ├── student_card_1.jpg
    │       ├── student_card_2.jpg
    │       └── ...
    └── others/
    ```

## 2. Huấn luyện mô hình
- Các kiến trúc mô hình dựa trên MobileNet, MobileNetV2, MobileNetV3-Small và MobileNetV3-Large.
- Thư viện TensorFlow được sử dụng để huấn luyện mô hình.
- Quá trình huấn luyện sẽ được thực hiện trên nền tảng [Kaggle](https://www.kaggle.com/), notebook huấn luyện [https://www.kaggle.com/code/sorasama/card-classification](https://www.kaggle.com/code/sorasama/card-classification)