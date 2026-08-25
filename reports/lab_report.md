# Báo cáo Lab Day 08

## 1. Thông tin sinh viên

- Họ tên: Nguyen Hoang Tin
- Repo/commit: local working tree
- Ngày thực hiện: 2026-08-25

## 2. Kiến trúc hệ thống

Workflow chuẩn hóa yêu cầu, phân loại bằng structured output của LLM, sau đó
định tuyến đến câu trả lời đơn giản, tra cứu bằng tool, yêu cầu bổ sung thông
tin hoặc luồng phê duyệt hành động rủi ro. Kết quả tool đi qua node đánh giá.
Node đánh giá ưu tiên LLM-as-judge có structured output và dùng heuristic an
toàn khi provider lỗi. Node retry có giới hạn sẽ thử lại tool hoặc chuyển yêu
cầu đến dead-letter handler. Mọi nhánh đều kết thúc tại `finalize → END`.

## 3. Schema của state

Các trường vô hướng (`route`, `attempt`, `evaluation_result`, thông tin
approval/action và kết quả cuối) được node mới nhất ghi đè. Các trường
`messages`, `tool_results`, `errors` và `events` dùng append reducer để giữ đầy
đủ lịch sử audit trong các lần retry. State có thể serialize thành JSON và mỗi
run sử dụng `thread_id` lấy từ scenario.

## 4. Kết quả các scenario

| Scenario | Route kỳ vọng | Route thực tế | Kết quả | Số lần retry | Số lần approval |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | Đạt | 0 | 0 |
| S02_tool | tool | tool | Đạt | 0 | 0 |
| S03_missing | missing_info | missing_info | Đạt | 0 | 0 |
| S04_risky | risky | risky | Đạt | 0 | 1 |
| S05_error | error | error | Đạt | 2 | 0 |
| S06_delete | risky | risky | Đạt | 0 | 1 |
| S07_dead_letter | error | error | Đạt | 1 | 0 |

Tổng cộng **7** scenario, tỷ lệ thành công **100.0%**,
**3** lần retry và **2** lần approval.
Đã ghi nhận state history: **Có**.

## 5. Phân tích lỗi và cơ chế an toàn

1. Lỗi tool được ghi rõ bằng kết quả `ERROR` và được đánh giá trước khi retry.
   Bộ đếm retry được so sánh với `max_attempts`, nên backend lỗi liên tục vẫn
   kết thúc an toàn tại dead-letter handler.
2. Refund, xóa dữ liệu, hủy dịch vụ, thay đổi tài khoản và gửi message ra ngoài
   được phân loại là risky. Các hành động này không thể chạy tool trước khi có
   approval; nếu bị từ chối, workflow chuyển sang yêu cầu bổ sung thông tin.
3. Thiếu cấu hình LLM được ghi thành error event và đi qua luồng lỗi có giới
   hạn, không tự tạo classification giả.

## 6. Persistence và khả năng khôi phục

Graph nhận checkpointer của LangGraph và mỗi lần invoke đều truyền
`configurable.thread_id`. Adapter hỗ trợ checkpoint in-memory và SQLite với
WAL mode. CLI kiểm tra state history của graph sau khi chạy và ghi kết quả vào
`resume_success` khi history tồn tại.

## 7. Phần mở rộng đã thực hiện

Đã triển khai SQLite persistence, LLM-as-judge và real HITL tùy chọn. Khi đặt
`LANGGRAPH_INTERRUPT=true`, approval node sử dụng cơ chế `interrupt()` của
LangGraph; mặc định mock approval giúp các lần chạy tự động ổn định.

## 8. Hướng cải thiện

Tích hợp support backend có xác thực và idempotency key, thay mock tool bằng
typed tool contract, đồng thời bổ sung tracing và đo latency cho từng node.
Approval trong môi trường production cũng nên yêu cầu reviewer đã xác thực và
có lệnh resume rõ ràng.
