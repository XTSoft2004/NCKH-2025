# Chuẩn bị tập dữ liệu

## 1. Gán nhãn dữ liệu
- Sử dụng công cụ **[LabelMe](https://github.com/wkentaro/labelme)** để gán nhãn các vùng chứa thông tin trên thẻ sinh viên. Mỗi ảnh sẽ được gán nhãn với các vùng chứa họ và tên, mã sinh viên, ngày sinh, lớp học và khóa học (hoặc thêm các thông tin khác nếu có).
- Mỗi ảnh gán nhãn sẽ được lưu trong thư mục cụ thể với định dạng JSON tương ứng.
- Cấu trúc thư mục tập dữ liệu đã được gán nhãn:
    ```
    datasets/
    ├── raw/ 
    │   ├── student_card_1.jpg
    │   ├── student_card_1.json
    │   ├── student_card_2.jpg
    │   ├── student_card_2.json
    │   └── ... 
    └── others/
    ```

## 2. Chia tập dữ liệu
- Sử dụng script `data_split.py` để chia tập dữ liệu thành các tập huấn luyện và kiểm tra. Script này sẽ đọc các file JSON đã gán nhãn, chuyển đổi định dạng sang định dạng PaddleOCR và lưu các file ảnh tương ứng vào các thư mục `train` và `test`.
- Các tham số của script:
  - `--input_dir`: Thư mục chứa các file ảnh và JSON gán nhãn.
  - `--output_dir`: Thư mục lưu trữ các tập dữ liệu đã chia.
  - `--train_ratio`: Tỷ lệ phần trăm của tập huấn luyện so với tổng số ảnh. Mặc định là 0.8 (80% cho tập huấn luyện, 20% cho tập kiểm tra).
- Ví dụ sử dụng:
  ```bash
  python datasets/data_split.py \
        --input_dir datasets/raw \
        --output_dir datasets/split \
        --train_ratio 0.8
  ```
- Kết quả sẽ được lưu trong thư mục `datasets/split` với cấu trúc như sau:
    ```
    datasets/
    ├── split/
    │   ├── train_images/
    │   ├── test_images/
    │   ├── train_labels.txt
    │   └── test_labels.txt
    └── others/
    ```

## 3. Tăng cường dữ liệu
- Sử dụng các kỹ thuật tăng cường dữ liệu như xoay, lật, thay đổi độ sáng, độ tương phản để tạo ra các biến thể của ảnh thẻ sinh viên. Điều này giúp mô hình học được các đặc điểm khác nhau của thẻ sinh viên trong các điều kiện khác nhau.
- Sử dụng script `data_augment.py` để tự động tăng cường dữ liệu với các tham số tùy chỉnh như sau:
  - `--input_dir`: Thư mục chứa ảnh gốc.
  - `--output_dir`: Thư mục lưu ảnh đã tăng cường.
  - `--augmentation_config`: Tập tin cấu hình YAML chứa các tham số tăng cường. File này được sinh ra bằng cách sử dụng `albumentations.save` để lưu các biến thể tăng cường (xem trong [Serializing an augmentation pipeline to a JSON or YAML file](https://albumentations.ai/docs/examples/serialization/#serializing-an-augmentation-pipeline-to-a-json-or-yaml-file))
  - `--augmentation_radio`: Số lượng ảnh tăng cường cần tạo cho mỗi ảnh gốc. Nếu tham số này là x thì sẽ tạo ra x ảnh tăng cường cho mỗi ảnh gốc, tổng số ảnh sau khi tăng cường sẽ là (x + 1) * số lượng ảnh gốc.  (Mặc định là 3)
  
> [!CAUTION]
Script trên chỉ được sử dụng để tăng cường dữ liệu với các loại tăng cường không làm thay đổi kích thước, vị trí của ảnh. Nếu sử dụng các loại tăng cường làm thay đổi kích thước, vị trí của ảnh thì cần phải cập nhật lại các nhãn tương ứng trong file JSON. 