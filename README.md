# Đồ án XLA – nhận diện mã QR và Data Matrix

Một chương trình chính: **Do_an_XLA.py**. Đọc một ảnh hoặc quét đệ quy thư mục bằng ZXing, WeChatQRCode và OpenCV. Dataset và model có sẵn được giữ để thử nghiệm offline.

## Cài đặt trên Windows

Python 3.10 trở lên; cấu hình đã kiểm thử bằng Python 3.10. Dùng môi trường riêng và phiên bản trong requirements.txt để tránh xung đột các gói OpenCV:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Không cần kích hoạt môi trường hay sửa ExecutionPolicy của PowerShell. Trên Linux/macOS, dùng .venv/bin/python.

## Chạy nhanh

Mặc định chạy không mở cửa sổ, không chờ nhấn phím:

```powershell
.\.venv\Scripts\python.exe Do_an_XLA.py --input qrcodes/decoding
.\.venv\Scripts\python.exe Do_an_XLA.py --input qrcodes/detection/demo --output out_qr/demo.json
```

Xem ảnh gốc và vùng mã được phát hiện:

```powershell
.\.venv\Scripts\python.exe Do_an_XLA.py --input qrcodes/detection/demo --display --pause
```

Nhấn phím trong cửa sổ OpenCV để tiếp tục, Escape để dừng khi dùng --pause.

| Tham số | Ý nghĩa |
| --- | --- |
| --input PATH | Một ảnh hoặc thư mục ảnh; bắt buộc |
| --display | Hiển thị ảnh và polygon kết quả |
| --no-display | Không GUI; mặc định |
| --show-steps, --show_steps | Bật cửa sổ tiền xử lý cho ảnh cần các bước bổ sung |
| --pause | Bật hiển thị, dừng sau mỗi ảnh |
| --time-budget N | Giới hạn mềm mỗi ảnh, mặc định 8 giây; 0 thử toàn bộ |
| --max-side N | Cạnh lớn nhất của ảnh làm việc ban đầu, mặc định 1600 |
| --limit N | Chỉ xử lý N ảnh đầu trong danh sách sắp theo tên |
| --output PATH.json | Lưu báo cáo vào một file mới |
| --strong-only | Tắt hai engine bổ sung ZBar/libdmtx |
| --model-dir PATH, --model_dir PATH | Thư mục model; mặc định nằm cạnh mã nguồn |
| --download-models, --download_models | Tải các model thiếu/rỗng; cần mạng |
| --verbose | Bật log chẩn đoán |

Đầu vào hỗ trợ đường dẫn có dấu tiếng Việt và đuôi ảnh viết hoa. Báo cáo JSON chứa trạng thái, nội dung mã, engine, bốn góc trong **tọa độ ảnh gốc** và thời gian từng ảnh. File đầu ra phải chưa tồn tại để tránh ghi đè kết quả cũ.

Giới hạn thời gian được kiểm tra giữa các lần gọi engine. Một lần gọi native C++ đang chạy có thể vượt giới hạn; đây không phải hard timeout. Ảnh quá mờ, bị che hoặc hỏng nhiều có thể không giải mã được. Chế độ nhanh dừng ở lần thử đầu có kết quả sau khi gộp các engine, không bảo đảm tìm đủ mọi mã trong ảnh phức tạp.

## Pipeline và tối ưu

1. Đọc ảnh, chuẩn hóa kích thước rồi giải mã ảnh xám gốc trước. Ảnh rõ không cần chạy toàn bộ tiền xử lý.
2. Nếu chưa giải mã được, thử ROI nghi ngờ với biến đổi phối cảnh và bổ sung viền trắng.
3. Thử CLAHE, làm nét, giảm bóng, nhị phân hóa, gamma, phóng ảnh và xoay. Tạo ảnh phóng khi cần và tái sử dụng trong cùng mức phóng.
4. Kiểm tra hình học, loại kết quả trùng vùng và đưa tọa độ về ảnh đầu vào. Hai mã cùng nội dung ở hai vị trí vẫn được giữ.

Bộ lọc chấp nhận mã có một ký tự và mã chiếm gần toàn ảnh. Xoay ảnh mở rộng khung để không cắt QR ở mép. Sắp tứ giác theo góc quanh tâm, tránh trùng đỉnh ở góc 45°. Tiền xử lý không inpaint toàn bộ nền trắng của QR.

Model WeChat có trong wechat_models/. Khi thiếu hoặc không khởi tạo được, chương trình cảnh báo và dùng các engine còn lại. Chỉ tải mạng khi chỉ định --download-models; tải qua file tạm rồi đổi tên khi hoàn tất. Nguồn model: [WeChatCV/opencv_3rdparty](https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode).

ZBar (pyzbar) và libdmtx (pylibdmtx) là tùy chọn, có thể cần thêm thư viện native. ZXing đã hỗ trợ cả QR và Data Matrix trong cấu hình mặc định.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Kiểm thử tạo mã trong bộ nhớ: QR ngắn/xoay/đảo màu, nhiều mã cùng nội dung, Data Matrix, tọa độ ROI và ảnh gốc, ảnh lỗi, đường dẫn Unicode, báo cáo JSON và ngân sách thời gian. tests/test_xla.py là bộ kiểm thử, không phải phiên bản đồ án khác.

Đo thử ngày 12/09/2026 trên máy local, không GUI:

| Bộ ảnh | Ảnh có ít nhất một mã giải được | Thời gian | Giới hạn mềm/ảnh |
| --- | --- | --- | --- |
| qrcodes/decoding | 26/26 | 0,95 giây | 3 giây |
| qrcodes/detection/demo | 16/16 | 3,01 giây | 3 giây |
| 8 ảnh đầu theo tên trong Damaged | 3/8 | 15,26 giây | 3 giây |

Đây là tỷ lệ ảnh có kết quả, **không phải** độ chính xác so với nhãn chuẩn hay tỷ lệ tìm đủ mọi mã. Thời gian phụ thuộc máy và cấu hình.

## Các file của đồ án XLA

```text
Do_an_XLA.py          # chương trình chính duy nhất
requirements.txt     # phiên bản thư viện đã kiểm thử
tests/test_xla.py     # kiểm thử hồi quy
wechat_models/       # model WeChatQR
qrcodes/             # dataset và ảnh minh họa có sẵn
Damaged/             # ảnh đầu vào bổ sung
out_qr/              # báo cáo phát sinh, được Git bỏ qua
```

Các bản Do_an_XLA_.py, Do_an_XLA_ban_full.py, Do_an_XLA_toiua.py, bản sao Code&dataset và ZIP đã được loại khỏi bộ mã làm việc. Dataset gốc vẫn được giữ tại qrcodes/detection.
