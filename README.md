# Scene Text Recognition Project M06

## Giới thiệu
**Nhận dạng văn bản trong ảnh (Scene Text Recognition):** là bài toán ứng dụng các kỹ thuật xử lý ảnh và nhận dạng chữ viết để xác định các văn bản xuất hiện trong các bức ảnh chụp từ môi trường thực tế.
Một số ứng dụng thực tế: 
- Xử lý văn bản trong ảnh: nhận diện chữ trong tài liệu, biển hiệu,...
- Tìm kiếm thông tin: Nhận diện chữ trong ảnh trên internet để thu thập dữ liệu quan trọng.
- Tự động hóa quy trình: Nhận diện chữ trong ảnh để tự động hóa các công việc, ví dụ như xử lý đơn hàng, thanh toán,..

**Quy trình của bài toán gồm 2 giai đoạn chính:**
- Phát hiện chữ viết (Detector): Xác định vị trí các khối văn bản trong ảnh.
- Nhận diện chữ viết (Recognizer): Giải mã văn bản tại các vị trí đã được xác định

Trong project này sử dụng YOLOv11(cho Detector) và CRNN(cho Recognizer)
- Đầu vào: Một bức ảnh chứa văn bản
- Đầu ra: Tọa độ vị trí và nội dung văn bản trong ảnh