# Kế hoạch Cập nhật Logic Load Data (8:1:1) và Thêm Dropout cho DenseNet

**Mục tiêu**: Cập nhật mã nguồn trong `densenet_pretrained` để load dữ liệu từ 1 thư mục Fold cụ thể. Huấn luyện thực sự trên nền tảng **DenseNet121**, với tỷ lệ dữ liệu là **8:1:1** (8 phần Train, 1 phần Validation lấy từ `train_val`; dùng nguyên `test` thư mục gốc). Mức **Dropout được cấu hình 0.2** trước lớp phân loại, và có tích hợp **Learning Rate Scheduler** tối ưu cho quá trình train lên đến **100 epochs**.

## User Review Required

> [!IMPORTANT]
> - Bản kế hoạch này đã chốt lại những trao đổi của bạn: Chọn train trên 1 Fold, tỷ lệ 8:1:1, chạy thực tế DenseNet121, set Dropout = 0.2 và chạy Epoch = 100 với Scheduler mới (mình chọn CosineAnnealingLR thay vì ReduceLROnPlateau).
> - Bạn vui lòng kiểm tra xem mình có đang chọn cấu hình scheduler hợp lý cho quy trình train 100 epochs của bạn chưa (ví dụ CosineAnnealingLR sẽ là sự lựa chọn tối ưu, mượt mà từ cao xuống thấp). 
> - Xác nhận kế hoạch cuối cùng để mình tiến hành lập trình.

## Proposed Changes

### 1. Component `data_setup.py`

#### [MODIFY] [data_setup.py](file:///e:/PROJECTWORSHOP/Eggplant_regconition/code_file/densenet_pretrained/data_setup.py)
Thay đổi hàm `create_dataloaders` đọc dữ liệu theo chuẩn Fold vật lý:
- Tham số truyền vào `data_dir` sẽ trỏ tới đường dẫn ví dụ `Fold_1`.
- Quét các ảnh thuộc thư mục `train_val`, sử dụng `train_test_split` (tỷ lệ Val = `1/9`, stratify theo nhãn) để bảo đảm đúng 10% bộ nguyên gốc thành Val và 80% thành Train.
- Quét trực tiếp thư mục `test` để ra tập Test 10% cuối cùng.
- Xóa tham số cấn `split_ratio` của phiên bản cũ. Trả về `train_loader`, `val_loader`, `test_loader`.
- Sửa phần chạy Unit Test (nếu file chạy riêng) dưới đáy script để mock phù hợp cấu trúc Fold.

### 2. Component `model_setup.py`

#### [MODIFY] [model_setup.py](file:///e:/PROJECTWORSHOP/Eggplant_regconition/code_file/densenet_pretrained/model_setup.py)
Chuẩn hoá mô hình về gốc DenseNet121 và thêm tính năng chống Overfitting:
- Sạch sẽ thay tên hàm `create_densenet169` thành `create_densenet121`.
- Đổi các object khởi tạo class từ `densenet169` -> `densenet121` (và tương đương cấu hình pretrained weight `DenseNet121_Weights`).
- Classifier cuối cùng sẽ được thay thế làm 1 chuỗi ngầm (Sequential) có Dropout.
```python
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_features, num_classes)
    )
```

### 3. Component `train.py`

#### [MODIFY] [train.py](file:///e:/PROJECTWORSHOP/Eggplant_regconition/code_file/densenet_pretrained/train.py)
Tuỳ chỉnh Training Pipeline cho kịch bản chạy dài (100 Epochs):
- Đổi tham số argparse `--epochs` mặc định lên **100**.
- Đổi tham số argparse `--data_dir` nhằm dẫn chứng hướng vào một thư mục đại diện (ví dụ `E:\PROJECTWORSHOP\Eggplant_regconition\10_Fold_CV_Dataset\Fold_1`).
- Thêm `--dropout_rate` với mặc định là **0.2**.
- **Cập nhật Scheduler**: Bỏ `ReduceLROnPlateau`, thay bằng `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)`. Đây là kĩ thuật xuống dốc học tập (learning decay) cực tốt trải dài theo tổng số lượng epochs giới hạn.

## Verification Plan

### Automated Tests
- Chạy khởi lệnh giả `python data_setup.py` để verify phần ruột load ra đủ Dataloaders cho Fold mock chưa, đếm output in ra coi chừng có lỗi không.
- Thử test chức năng của Dropouts bằng mã mẫu của `model_setup.py`.

### Manual Verification
Sau khi code xong, mình sẽ tạo một file hướng dẫn (Walkthrough) gửi bạn tham khảo, đồng thời bạn có thể tự mình chạy lệnh huấn luyện `python train.py --epochs 2` trên máy tính windows cục bộ của bạn để đánh giá tiến độ trước khi đẩy 100 epochs chính thức.
