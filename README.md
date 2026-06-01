# BÀI TẬP THỰC HÀNH: XÂY DỰNG MẠNG NƠ-RON TÍCH CHẬP (CNN) VỚI TENSORFLOW

[cite_start]Dự án này tập trung vào việc áp dụng Mạng nơ-ron tích chập (Convolutional Neural Network - CNN) để giải quyết các bài toán thị giác máy tính, cụ thể là nhận dạng và phân loại đối tượng trong ảnh[cite: 6, 7].

## 📊 Tập dữ liệu (Dataset)

Dự án thực hành trên đa dạng các tập dữ liệu hình ảnh:
* [cite_start]**MNIST Digit (File CSV):** Tập dữ liệu nhận dạng ảnh chữ viết tay các số từ 0 đến 9[cite: 320, 321].
* [cite_start]**CIFAR10:** Tập ảnh phân loại 10 đối tượng (máy bay, ô tô, chim, mèo, chó, ếch, ngựa, tàu, xe tải)[cite: 557, 558].
* [cite_start]**Dogs vs Cats:** Tập dữ liệu hình ảnh để phân loại chó và mèo[cite: 559]. Đã tích hợp script quét và loại bỏ các file ảnh hỏng (corrupted JFIF).
* [cite_start]**Fashion MNIST:** Tập dữ liệu hình ảnh chứa các mẫu quần áo với 10 nhãn phân loại[cite: 561].
* [cite_start]**Men/Women Classification:** Tập dữ liệu phân loại khuôn mặt nam và nữ[cite: 562]. Kèm theo script tự động dọn dẹp, chuẩn hóa cấu trúc thư mục từ file nén thô.

## 🛠️ Quy trình thực hiện chi tiết

### 1. Tiền xử lý và Làm sạch dữ liệu (Data Cleaning & Preprocessing)
* Tự động quét và xóa các file hình ảnh bị lỗi cấu trúc bit (không chuẩn JFIF) để tránh gây sập mô hình trong quá trình huấn luyện.
* Gom nhóm và thiết lập cấu trúc thư mục tự động bằng Python script đối với các bộ dữ liệu thô.
* [cite_start]Đọc dữ liệu ảnh và biến đổi cấu trúc (Reshape) để phù hợp với Input Shape của mô hình CNN (ví dụ: `28x28x1` hoặc `150x150x3`)[cite: 335, 339].
* [cite_start]Chuẩn hóa dữ liệu pixel ảnh bằng cách chia tỷ lệ cho 255.0 (`MinMaxScaler` tương đương)[cite: 336, 337].
* [cite_start]Chuyển đổi nhãn (labels) sang định dạng One-hot encoding đối với bài toán phân loại nhiều lớp[cite: 405, 406].
* Tự động phân chia tập kiểm định (Validation Split = 20%) trực tiếp qua hàm `image_dataset_from_directory`.

### 2. Trực quan hóa dữ liệu (Data Visualization)
* [cite_start]Sử dụng thư viện `matplotlib` để hiển thị trực quan các ảnh đầu vào cùng với nhãn tương ứng[cite: 346, 347, 348].
* [cite_start]Vẽ đồ thị phân tích sự biến thiên của độ chính xác (Accuracy) và hàm mất mát (Loss) trên tập huấn luyện và tập validation qua từng epoch để theo dõi hiệu suất[cite: 433, 434, 464, 465].

### 3. Xây dựng và Huấn luyện mô hình CNN
* [cite_start]Khởi tạo kiến trúc mạng `Sequential` sử dụng Keras/TensorFlow[cite: 413].
* [cite_start]Thêm các tầng chập (Convolutional Layer) với hàm kích hoạt ReLU để rút trích đặc trưng ảnh và tính non-linearity[cite: 221, 284].
* [cite_start]Thêm các tầng Max Pooling để giảm chiều dữ liệu, giảm tính toán và hạn chế overfitting[cite: 287].
* [cite_start]Làm phẳng ma trận (`Flatten`) và kết nối với các tầng kết nối đầy đủ (`Dense` / Fully Connected Layer)[cite: 419, 420].
* [cite_start]Sử dụng hàm `softmax` cho bài toán nhiều lớp (MNIST, CIFAR10) [cite: 306] và hàm `sigmoid` cho bài toán phân loại nhị phân (Chó/Mèo, Nam/Nữ).
* [cite_start]Thiết lập thuật toán tối ưu `adam` và hàm mất mát (`categorical_crossentropy` hoặc `binary_crossentropy`)[cite: 429].
* Huấn luyện mô hình và lưu trọng số mạng nơ-ron dưới định dạng chuẩn `.keras`.

### 4. Đánh giá và Dự báo (Testing)
* [cite_start]Đánh giá hiệu suất mô hình trên tập kiểm thử (Testing set) thông qua hàm `evaluate`[cite: 487, 488].
* Thiết kế mã nguồn dự báo tách biệt hoàn toàn với mã huấn luyện. [cite_start]Hệ thống nạp lại mô hình (`load_model`) để nhận diện ảnh tải lên theo thời gian thực mà không cần train lại[cite: 522, 532].

### 5. Triển khai ứng dụng Web (Deployment)
* [cite_start]Đóng gói mô hình nhận diện CNN (ví dụ: CIFAR10) và triển khai lên nền tảng Web sử dụng framework Flask[cite: 564].
* Cung cấp giao diện cho phép người dùng upload ảnh trực tiếp và nhận kết quả phân loại nhanh chóng. Tích hợp public URL thông qua `pyngrok` đối với môi trường máy ảo.

## 🚀 Hướng dẫn chạy mã nguồn (Google Colab & IDLE)

Để đảm bảo tính linh hoạt và gọn gàng, hệ thống cung cấp mã nguồn tương thích tuyệt đối cho hai môi trường (với mã Train và Test được tách biệt):

**Tùy chọn 1: Chạy trên Google Colab**
1. Đóng gói tập dữ liệu ảnh thành file nén `.zip` trên máy tính.
2. Tải file `.zip` lên phần Tệp của Colab. **Lưu ý quan trọng:** Chờ vòng tròn tiến trình ở góc dưới bên trái hoàn tất 100% trước khi thao tác tiếp.
3. Chạy lệnh giải nén `!unzip -q ten_file.zip` và các script dọn rác dữ liệu đi kèm.
4. Chạy ô lệnh Huấn luyện để tạo file `.keras`.
5. Chạy ô lệnh Dự báo. Lệnh `files.upload()` sẽ hiển thị nút tải ảnh lên để kiểm tra mô hình.
