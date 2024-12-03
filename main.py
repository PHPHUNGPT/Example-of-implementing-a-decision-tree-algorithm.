# Nhập các thư viện cần thiết
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, filedialog, Toplevel, Canvas, Scrollbar
from PIL import Image, ImageTk  # Thư viện Pillow để xử lý hình ảnh
from graphviz import Digraph
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Hàm tính entropy của một tập dữ liệu
def entropy(y):
    value, counts = np.unique(y, return_counts=True)  # Tìm các giá trị duy nhất và đếm số lượng của chúng
    probabilities = counts / len(y)  # Tính xác suất của từng giá trị
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])  # Tính entropy dựa trên công thức
    return entropy  # Trả về giá trị entropy

# Hàm tìm ngưỡng tốt nhất để chia dữ liệu liên tục
def best_split(data, attribute, target_name="class"):
    unique_values = np.sort(data[attribute].unique())  # Lấy các giá trị duy nhất của thuộc tính và sắp xếp chúng
    best_gain = -1  # Khởi tạo độ lợi thông tin tốt nhất
    best_threshold = None  # Khởi tạo ngưỡng tốt nhất
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2  # Tính ngưỡng trung bình giữa hai giá trị liên tiếp
        data_below = data[data[attribute] <= threshold]  # Dữ liệu dưới ngưỡng
        data_above = data[data[attribute] > threshold]  # Dữ liệu trên ngưỡng
        current_gain = info_gain_continuous(data, attribute, threshold, target_name)  # Tính độ lợi thông tin cho ngưỡng này
        if current_gain > best_gain:  # Nếu độ lợi thông tin hiện tại tốt hơn độ lợi tốt nhất trước đó
            best_gain = current_gain  # Cập nhật độ lợi thông tin tốt nhất
            best_threshold = threshold  # Cập nhật ngưỡng tốt nhất
    return best_threshold, best_gain  # Trả về ngưỡng tốt nhất và độ lợi thông tin tốt nhất

# Hàm tính độ lợi thông tin cho thuộc tính liên tục
def info_gain_continuous(data, attribute, threshold, target_name="class"):
    total_entropy = entropy(data[target_name])  # Tính entropy tổng thể của dữ liệu
    data_below = data[data[attribute] <= threshold]  # Dữ liệu dưới ngưỡng
    data_above = data[data[attribute] > threshold]  # Dữ liệu trên ngưỡng
    weight_below = len(data_below) / len(data)  # Trọng số của dữ liệu dưới ngưỡng
    weight_above = len(data_above) / len(data)  # Trọng số của dữ liệu trên ngưỡng
    entropy_below = entropy(data_below[target_name])  # Tính entropy của dữ liệu dưới ngưỡng
    entropy_above = entropy(data_above[target_name])  # Tính entropy của dữ liệu trên ngưỡng
    weighted_entropy = (weight_below * entropy_below) + (weight_above * entropy_above)  # Tính entropy trung bình trọng số
    information_gain = total_entropy - weighted_entropy  # Tính độ lợi thông tin
    return information_gain  # Trả về độ lợi thông tin

# Hàm tính độ lợi thông tin (information gain) của một thuộc tính
def info_gain(data, split_attribute_name, target_name="class"):
    if data[split_attribute_name].dtype.kind in 'bifc':  # Kiểm tra xem thuộc tính có phải là liên tục không
        threshold, gain = best_split(data, split_attribute_name, target_name)  # Tìm ngưỡng tốt nhất và độ lợi thông tin
        return gain, threshold  # Trả về độ lợi thông tin và ngưỡng
    else:  # Nếu thuộc tính là dạng danh mục
        total_entropy = entropy(data[target_name])  # Tính entropy tổng thể của dữ liệu
        values, counts = np.unique(data[split_attribute_name], return_counts=True)  # Tìm các giá trị duy nhất và đếm số lượng của chúng
        weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data[data[split_attribute_name] == values[i]][target_name]) for i in range(len(values))])  # Tính entropy trung bình trọng số cho từng giá trị
        information_gain = total_entropy - weighted_entropy  # Tính độ lợi thông tin
        return information_gain, None  # Trả về độ lợi thông tin và None cho ngưỡng (không có ngưỡng cho thuộc tính danh mục)

# Hàm tạo cây quyết định ID3
def create_tree_id3(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:  # Nếu tất cả các mục tiêu là giống nhau
        return np.unique(data[target_attribute_name])[0]  # Trả về giá trị mục tiêu đó
    elif len(data) == 0:  # Nếu dữ liệu rỗng
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]  # Trả về giá trị mục tiêu phổ biến nhất trong dữ liệu ban đầu
    elif len(features) == 0:  # Nếu không còn thuộc tính nào để chia
        return parent_node_class  # Trả về lớp của nút cha
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]  # Lớp phổ biến nhất
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]  # Tính độ lợi thông tin cho từng thuộc tính
        best_feature_index = np.argmax([item[0] for item in item_values])  # Tìm thuộc tính có độ lợi thông tin cao nhất
        best_feature, threshold = item_values[best_feature_index]  # Thuộc tính và ngưỡng tốt nhất
        tree = {features[best_feature_index]: {}}  # Khởi tạo cây với thuộc tính tốt nhất
        best_feature_name = features[best_feature_index]  # Tên của thuộc tính tốt nhất
        features = [i for i in features if i != best_feature_name]  # Loại bỏ thuộc tính tốt nhất khỏi danh sách thuộc tính
        
        if threshold is not None:  # Nếu thuộc tính là liên tục
            data_below = data[data[best_feature_name] <= threshold]  # Dữ liệu dưới ngưỡng
            data_above = data[data[best_feature_name] > threshold]  # Dữ liệu trên ngưỡng
            # Đảm bảo tập dữ liệu con không rỗng
            if not data_below.empty:
                subtree_below = create_tree_id3(data_below, data, features, target_attribute_name, parent_node_class)  # Tạo cây con cho dữ liệu dưới ngưỡng
                tree[best_feature_name][f"<= {threshold}"] = subtree_below  # Thêm cây con vào cây chính
            if not data_above.empty:
                subtree_above = create_tree_id3(data_above, data, features, target_attribute_name, parent_node_class)  # Tạo cây con cho dữ liệu trên ngưỡng
                tree[best_feature_name][f"> {threshold}"] = subtree_above  # Thêm cây con vào cây chính
        else:  # Nếu thuộc tính là danh mục
            for value in np.unique(data[best_feature_name]):
                sub_data = data[data[best_feature_name] == value]  # Dữ liệu con cho giá trị hiện tại
                if not sub_data.empty:  # Đảm bảo tập dữ liệu con không rỗng
                    subtree = create_tree_id3(sub_data, data, features, target_attribute_name, parent_node_class)  # Tạo cây con cho giá trị hiện tại
                    tree[best_feature_name][value] = subtree  # Thêm cây con vào cây chính
        return tree  # Trả về cây quyết định

# Hàm tạo cây quyết định C4.5
def create_tree_c45(data, original_data, features, target_attribute_name="class", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:  # Nếu tất cả các mục tiêu là giống nhau
        return np.unique(data[target_attribute_name])[0]  # Trả về giá trị mục tiêu đó
    elif len(data) == 0:  # Nếu dữ liệu rỗng
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]  # Trả về giá trị mục tiêu phổ biến nhất trong dữ liệu ban đầu
    elif len(features) == 0:  # Nếu không còn thuộc tính nào để chia
        return parent_node_class  # Trả về lớp của nút cha
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]  # Lớp phổ biến nhất
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]  # Tính độ lợi thông tin cho từng thuộc tính
        best_feature_index = np.argmax([item[0] for item in item_values])  # Tìm thuộc tính có độ lợi thông tin cao nhất
        best_feature, threshold = item_values[best_feature_index]  # Thuộc tính và ngưỡng tốt nhất
        tree = {features[best_feature_index]: {}}  # Khởi tạo cây với thuộc tính tốt nhất
        best_feature_name = features[best_feature_index]  # Tên của thuộc tính tốt nhất
        features = [i for i in features if i != best_feature_name]  # Loại bỏ thuộc tính tốt nhất khỏi danh sách thuộc tính
        
        if threshold is not None:  # Nếu thuộc tính là liên tục
            data_below = data[data[best_feature_name] <= threshold]  # Dữ liệu dưới ngưỡng
            data_above = data[data[best_feature_name] > threshold]  # Dữ liệu trên ngưỡng
            # Đảm bảo tập dữ liệu con không rỗng
            if not data_below.empty:
                subtree_below = create_tree_c45(data_below, data, features, target_attribute_name, parent_node_class)  # Tạo cây con cho dữ liệu dưới ngưỡng
                tree[best_feature_name][f"<= {threshold}"] = subtree_below  # Thêm cây con vào cây chính
            if not data_above.empty:
                subtree_above = create_tree_c45(data_above, data, features, target_attribute_name, parent_node_class)  # Tạo cây con cho dữ liệu trên ngưỡng
                tree[best_feature_name][f"> {threshold}"] = subtree_above  # Thêm cây con vào cây chính
        else:  # Nếu thuộc tính là danh mục
            for value in np.unique(data[best_feature_name]):
                sub_data = data[data[best_feature_name] == value]  # Dữ liệu con cho giá trị hiện tại
                if not sub_data.empty:  # Đảm bảo tập dữ liệu con không rỗng
                    subtree = create_tree_c45(sub_data, data, features, target_attribute_name, parent_node_class)  # Tạo cây con cho giá trị hiện tại
                    tree[best_feature_name][value] = subtree  # Thêm cây con vào cây chính
        return tree  # Trả về cây quyết định

# Hàm trực quan hóa cây quyết định
def visualize_tree(tree, graph=None, parent=None, edge_label=''):
    if graph is None:  # Nếu đồ thị chưa được khởi tạo
        graph = Digraph()  # Khởi tạo đồ thị mới
    if isinstance(tree, dict):  # Nếu nút hiện tại là từ điển (nút nội)
        root = next(iter(tree))  # Lấy thuộc tính gốc
        for k, v in tree[root].items():  # Duyệt qua các nhánh của cây
            node_id = f"{root}_{k}"  # Tạo ID cho nút
            graph.node(node_id, label=f"{root}={k}")  # Thêm nút vào đồ thị
            if parent:  # Nếu có nút cha
                graph.edge(parent, node_id, label=edge_label)  # Thêm cạnh vào đồ thị
            visualize_tree(v, graph, node_id)  # Gọi đệ quy để thêm các nút con
    else:  # Nếu nút hiện tại là lá (giá trị cuối)
        leaf_id = f"leaf_{tree}"  # Tạo ID cho lá
        graph.node(leaf_id, label=str(tree), shape='ellipse')  # Thêm lá vào đồ thị
        if parent:  # Nếu có nút cha
            graph.edge(parent, leaf_id, label=edge_label)  # Thêm cạnh vào đồ thị
    return graph  # Trả về đồ thị đã tạo

# Hàm tải dữ liệu từ file
def load_data():
    filepath = filedialog.askopenfilename()  # Mở hộp thoại để chọn file
    if filepath:  # Nếu người dùng chọn file
        try:
            data = pd.read_csv(filepath)  # Đọc dữ liệu từ file CSV
            if data.empty:  # Kiểm tra nếu dữ liệu rỗng
                messagebox.showerror("Error", "The selected file is empty.")  # Hiển thị thông báo lỗi
                return None  # Trả về None
            return data  # Trả về dữ liệu
        except Exception as e:  # Bắt lỗi nếu có
            messagebox.showerror("Error", f"Error reading the file: {e}")  # Hiển thị thông báo lỗi
            return None  # Trả về None
    messagebox.showwarning("Warning", "No file selected.")  # Hiển thị thông báo cảnh báo nếu không chọn file
    return None  # Trả về None

# Hàm huấn luyện mô hình
def train_model():
    data = load_data()  # Tải dữ liệu
    if data is not None:  # Nếu dữ liệu không rỗng
        target_attribute_name = tk.simpledialog.askstring("Input", "Please enter the name of the target column:")  # Yêu cầu nhập tên thuộc tính mục tiêu
        if target_attribute_name not in data.columns:  # Kiểm tra xem thuộc tính mục tiêu có trong dữ liệu không
            messagebox.showerror("Error", "Target column not found in data.")  # Hiển thị thông báo lỗi
            return  # Thoát hàm
        algorithm_choice = tk.simpledialog.askstring("Input", "Choose algorithm (C4.5 or ID3):")  # Yêu cầu chọn thuật toán
        if algorithm_choice not in ["C4.5", "ID3"]:  # Kiểm tra xem lựa chọn hợp lệ không
            messagebox.showerror("Error", "Invalid algorithm choice.")  # Hiển thị thông báo lỗi
            return  # Thoát hàm
        train_data, test_data = train_test_split(data, test_size=0.2)  # Chia dữ liệu thành tập huấn luyện và kiểm tra
        features = [col for col in train_data.columns if col != target_attribute_name]  # Lấy danh sách thuộc tính (trừ mục tiêu)
        if algorithm_choice == "C4.5":  # Nếu chọn thuật toán C4.5
            tree = create_tree_c45(train_data, train_data, features, target_attribute_name)  # Tạo cây quyết định C4.5
        else:  # Nếu chọn thuật toán ID3
            tree = create_tree_id3(train_data, train_data, features, target_attribute_name)  # Tạo cây quyết định ID3
        result_text.delete("1.0", tk.END)  # Xóa nội dung hiển thị kết quả cũ
        result_text.insert(tk.END, str(tree))  # Hiển thị cây quyết định
        visualize_and_display_tree(tree)  # Trực quan hóa và hiển thị cây
        accuracy = evaluate_model(tree, test_data, target_attribute_name)  # Đánh giá độ chính xác của mô hình
        messagebox.showinfo("Model Accuracy", f"Accuracy on test data: {accuracy:.2f}")  # Hiển thị độ chính xác của mô hình
    else:
        messagebox.showerror("Error", "Failed to load data.")  # Hiển thị thông báo lỗi nếu không tải được dữ liệu

# Hàm đánh giá mô hình
def evaluate_model(tree, test_data, target_attribute_name):
    correct_predictions = 0  # Khởi tạo số lượng dự đoán đúng
    for _, row in test_data.iterrows():  # Duyệt qua từng hàng trong dữ liệu kiểm tra
        prediction = classify(tree, row)  # Phân loại mẫu dữ liệu
        if prediction == row[target_attribute_name]:  # Nếu dự đoán đúng
            correct_predictions += 1  # Tăng số lượng dự đoán đúng
    return correct_predictions / len(test_data)  # Trả về độ chính xác

# Hàm phân loại mẫu dữ liệu
def classify(tree, row):
    if not isinstance(tree, dict):  # Nếu nút hiện tại không phải là từ điển (là lá)
        return tree  # Trả về giá trị của lá
    attribute = next(iter(tree))  # Lấy thuộc tính gốc
    for key in tree[attribute].keys():  # Duyệt qua các nhánh của cây
        if key.startswith("<="):  # Nếu là ngưỡng <=
            threshold = float(key.split("<= ")[1])  # Tách và chuyển đổi ngưỡng thành số
            if row[attribute] <= threshold:  # Nếu giá trị thuộc tính nhỏ hơn hoặc bằng ngưỡng
                return classify(tree[attribute][key], row)  # Phân loại dữ liệu dưới ngưỡng
        elif key.startswith(">"):  # Nếu là ngưỡng >
            threshold = float(key.split("> ")[1])  # Tách và chuyển đổi ngưỡng thành số
            if row[attribute] > threshold:  # Nếu giá trị thuộc tính lớn hơn ngưỡng
                return classify(tree[attribute][key], row)  # Phân loại dữ liệu trên ngưỡng
        elif row[attribute] == key:  # Nếu giá trị thuộc tính khớp với nhánh
            return classify(tree[attribute][key], row)  # Phân loại dữ liệu cho giá trị khớp
    return None  # Trả về None nếu không có nhánh phù hợp

# Hàm trực quan hóa và hiển thị cây quyết định
def visualize_and_display_tree(tree):
    graph = visualize_tree(tree)  # Tạo đồ thị cho cây quyết định
    graph.render('decision_tree', format='png', cleanup=True)  # Tạo tệp PNG từ đồ thị
    display_tree_image('decision_tree.png')  # Hiển thị hình ảnh cây quyết định

# Hàm hiển thị hình ảnh cây quyết định
def display_tree_image(image_path):
    top = Toplevel()  # Tạo cửa sổ con mới
    top.title("Decision Tree Visualization")  # Đặt tiêu đề cho cửa sổ con

    canvas = Canvas(top)  # Tạo vùng vẽ cho cửa sổ con
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Đặt vùng vẽ vào cửa sổ con

    scrollbar = Scrollbar(top, orient=tk.VERTICAL, command=canvas.yview)  # Tạo thanh cuộn dọc
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Đặt thanh cuộn vào cửa sổ con
    canvas.config(yscrollcommand=scrollbar.set)  # Liên kết thanh cuộn với vùng vẽ

    h_scrollbar = Scrollbar(top, orient=tk.HORIZONTAL, command=canvas.xview)  # Tạo thanh cuộn ngang
    h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)  # Đặt thanh cuộn vào cửa sổ con
    canvas.config(xscrollcommand=h_scrollbar.set)  # Liên kết thanh cuộn với vùng vẽ

    img = Image.open(image_path)  # Mở hình ảnh từ tệp
    img_tk = ImageTk.PhotoImage(img)  # Chuyển đổi hình ảnh thành đối tượng PhotoImage
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)  # Thêm hình ảnh vào vùng vẽ
    canvas.image = img_tk  # Giữ tham chiếu đến hình ảnh để không bị thu hồi

    canvas.config(scrollregion=canvas.bbox(tk.ALL))  # Đặt vùng cuộn để bao gồm toàn bộ hình ảnh

# Hàm lưu cây quyết định vào file
def save_tree():
    tree_text = result_text.get("1.0", tk.END).strip()  # Lấy nội dung cây từ vùng hiển thị kết quả
    if tree_text:  # Nếu có nội dung cây
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])  # Mở hộp thoại để chọn nơi lưu file
        if filepath:  # Nếu người dùng chọn nơi lưu
            try:
                with open(filepath, "w") as f:  # Mở file để ghi
                    f.write(tree_text)  # Ghi nội dung cây vào file
                messagebox.showinfo("Success", "Tree saved successfully.")  # Hiển thị thông báo thành công
            except Exception as e:  # Bắt lỗi nếu có
                messagebox.showerror("Error", f"Error saving the file: {e}")  # Hiển thị thông báo lỗi
    else:
        messagebox.showwarning("Warning", "No tree to save.")  # Hiển thị thông báo cảnh báo nếu không có cây để lưu

# Tạo giao diện ứng dụng
app = tk.Tk()  # Tạo cửa sổ chính của ứng dụng
app.title("Decision Tree Classifier")  # Đặt tiêu đề cho cửa sổ chính

# Nút để tải dữ liệu và huấn luyện mô hình
load_button = tk.Button(app, text="Load Data and Train Model", command=train_model)  # Tạo nút để tải dữ liệu và huấn luyện mô hình
load_button.pack(pady=10)  # Đặt nút vào cửa sổ chính với khoảng đệm

# Nút để lưu cây quyết định
save_button = tk.Button(app, text="Save Tree", command=save_tree)  # Tạo nút để lưu cây quyết định
save_button.pack(pady=10)  # Đặt nút vào cửa sổ chính với khoảng đệm

# Vùng hiển thị kết quả
result_text = tk.Text(app, height=10, width=80)  # Tạo vùng văn bản để hiển thị kết quả
result_text.pack(pady=10)  # Đặt vùng văn bản vào cửa sổ chính với khoảng đệm

# Vùng vẽ cây quyết định
canvas = tk.Canvas(app, width=800, height=600)  # Tạo vùng vẽ để hiển thị cây quyết định
canvas.pack(pady=10)  # Đặt vùng vẽ vào cửa sổ chính với khoảng đệm

# Khởi động ứng dụng
app.mainloop()  # Bắt đầu vòng lặp sự kiện của ứng dụng
