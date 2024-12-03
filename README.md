# C4.5 Decision Tree Classifier

Ứng dụng này triển khai một bộ phân loại cây quyết định sử dụng thuật toán C4.5. Ứng dụng cho phép người dùng tải dữ liệu, huấn luyện mô hình cây quyết định, trực quan hóa cây quyết định, và lưu cây quyết định vào file.

## Yêu cầu hệ thống

- Python 3.x
- Các thư viện Python:
  - numpy
  - pandas
  - tkinter
  - graphviz
  - scikit-learn

## Cài đặt

1. Tải và cài đặt [Python](https://www.python.org/downloads/).

2. Cài đặt các thư viện cần thiết bằng cách chạy lệnh sau trong terminal hoặc command prompt:
    ```bash
    pip install numpy pandas tkinter graphviz scikit-learn
    ```

3. Đảm bảo rằng bạn đã cài đặt Graphviz. Bạn có thể tải và cài đặt Graphviz từ [Graphviz download page](https://graphviz.gitlab.io/download/).

## Hướng dẫn sử dụng

1. Tải mã nguồn về máy tính của bạn.

2. Chạy tệp `main.py` bằng lệnh sau:
    ```bash
    python main.py
    ```

3. Giao diện ứng dụng sẽ xuất hiện với các chức năng sau:

    - **Load Data and Train Model**: Nhấn nút này để chọn tệp dữ liệu CSV và huấn luyện mô hình cây quyết định. Sau khi huấn luyện, cây quyết định sẽ được hiển thị dưới dạng văn bản và đồ thị.
    - **Save Tree**: Nhấn nút này để lưu cây quyết định vào tệp văn bản.

## Mô tả mã nguồn

### Các hàm chính

- `entropy(y)`: Tính entropy của tập dữ liệu.
- `info_gain(data, split_attribute_name, target_name="class")`: Tính độ lợi thông tin của một thuộc tính.
- `create_tree(data, original_data, features, target_attribute_name="class", parent_node_class=None)`: Tạo cây quyết định từ dữ liệu huấn luyện.
- `visualize_tree(tree, graph=None, parent=None, edge_label='')`: Trực quan hóa cây quyết định sử dụng Graphviz.
- `load_data()`: Tải dữ liệu từ tệp CSV.
- `train_model()`: Huấn luyện mô hình cây quyết định từ dữ liệu được tải.
- `evaluate_model(tree, test_data, target_attribute_name)`: Đánh giá mô hình cây quyết định trên tập dữ liệu kiểm tra.
- `classify(tree, row)`: Phân loại một mẫu dữ liệu sử dụng cây quyết định.
- `visualize_and_display_tree(tree)`: Trực quan hóa và hiển thị cây quyết định.
- `save_tree()`: Lưu cây quyết định vào tệp văn bản.

### Giao diện người dùng

- Sử dụng thư viện `tkinter` để tạo giao diện người dùng.
- Giao diện bao gồm các nút bấm để tải dữ liệu, huấn luyện mô hình và lưu cây quyết định, cũng như vùng hiển thị kết quả và canvas để vẽ cây quyết định.

## Ghi chú

- Đảm bảo rằng tệp dữ liệu CSV phải có cột mục tiêu (target column) để huấn luyện mô hình cây quyết định.
- Đảm bảo rằng thư viện Graphviz đã được cài đặt đúng cách trên hệ thống để trực quan hóa cây quyết định.

## Tác giả

- Tên: [Tên của bạn]
- Email: [Email của bạn]

