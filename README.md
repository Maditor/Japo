# Japo

**Tác giả:** [Maditor](https://github.com/Maditor) · **Repo:** [github.com/Maditor/Japo](https://github.com/Maditor/Japo)

Phụ đề **thời gian thực** cho video tiếng Nhật trên Windows.
Japo nghe âm thanh đang phát trên máy (YouTube, trình phát phim, trình duyệt…), nhận diện lời thoại bằng **Whisper**, dịch sang **tiếng Việt** và hiện từng câu trong một thanh dọc gọn gàng bên cạnh màn hình.

- Không cần file phụ đề, không cần tải video về
- Dịch bằng AI (Cloudflare Workers AI / Gemini), có ngữ cảnh nên xưng hô tự nhiên
- Hiện kèm phiên âm **romaji** và bản **tiếng Anh**
- Đoán **giọng nam/nữ** để xưng hô đúng hơn
- Giao diện sáng/tối, nền trong suốt, luôn nằm trên cùng, nhớ vị trí cửa sổ

---

## 1. Yêu cầu trước khi cài

| Thứ cần có | Bắt buộc? | Ghi chú |
|---|---|---|
| **Windows 10 hoặc 11 (64-bit)** | Có | Japo dùng tính năng thu âm hệ thống của Windows |
| **Python 3.11** | Có | Bản 3.10 và 3.12 cũng chạy được. Khi cài **phải tick "Add Python to PATH"** |
| **Internet** | Có | Lần đầu cài tải khoảng **2–3 GB** thư viện, lần đầu chạy tải thêm model khoảng **1.6 GB** |
| **Ổ cứng trống ~6 GB** | Có | Thư viện ~3 GB + model ~1.6 GB + bản đóng gói (nếu build) |
| **Card NVIDIA + driver mới** | Nên có | Chạy nhanh hơn nhiều. Không có card NVIDIA thì Japo tự chạy bằng CPU (chậm hơn) |
| **Tài khoản Cloudflare** (miễn phí) | Nên có | Để dịch bằng AI. Không có thì Japo dùng máy dịch Microsoft/Google (kém tự nhiên hơn) |
| **Gemini API key** | Tuỳ chọn | Dự phòng khi Cloudflare lỗi |

### Cài Python nhanh bằng lệnh
Mở **cmd** và chạy:
```
winget install -e --id Python.Python.3.11 --override "/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1"
```
Cài xong **đóng cmd, mở lại** rồi gõ `python --version`, thấy `Python 3.11.x` là được.

> Nếu gõ `python` mà Microsoft Store tự mở ra: vào *Settings → Apps → Advanced app settings → App execution aliases* và tắt 2 mục **python.exe**, **python3.exe**.

---

## 2. Cài đặt

1. **Tải source:** vào [github.com/Maditor/Japo](https://github.com/Maditor/Japo), bấm nút **Code → Download ZIP** rồi giải nén. Hoặc dùng lệnh:
   ```
   git clone https://github.com/Maditor/Japo.git
   ```
   Nên để ở chỗ cố định, ví dụ `D:\Apps\Japo`, và **đừng** để trong `C:\Program Files`.
2. Bấm đúp **`setup.bat`** rồi chờ nó tải và cài thư viện (khá lâu ở lần đầu).
   Thấy dòng `Setup complete` là xong.
3. *(Nên làm)* Cấu hình dịch bằng AI, xem mục 3.
4. Bấm đúp **`run.bat`** để mở Japo.
   Lần đầu mở, Japo tải model nhận diện giọng nói (~1.6 GB). Chờ đến khi thanh dưới cùng hiện **Listening** là dùng được.

> **Tải chậm?** Hai gói `nvidia-cublas-cu12` và `nvidia-cudnn-cu12` rất nặng (~1.3 GB). Nếu pip tải quá chậm, bạn có thể tải file `.whl` của chúng trên [pypi.org](https://pypi.org) bằng trình duyệt hoặc IDM (chọn bản `win_amd64`), rồi cài bằng lệnh
> `venv\Scripts\python.exe -m pip install tên-file.whl`, sau đó chạy lại `setup.bat`.

---

## 3. Cấu hình dịch bằng AI (Cloudflare)

1. Đăng nhập [dash.cloudflare.com](https://dash.cloudflare.com) (tạo tài khoản miễn phí nếu chưa có).
2. Lấy **Account ID**: nằm trong trang *Workers AI*, hoặc là dãy 32 ký tự trên thanh địa chỉ ngay sau `dash.cloudflare.com/`.
3. Tạo **API Token**: vào [trang API Tokens](https://dash.cloudflare.com/profile/api-tokens) → **Create Token** → chọn mẫu **Workers AI** → tạo và copy token.
   Chú ý: phải là **API Token**, không phải *Global API Key*.
4. Mở Japo, bấm **⚙ → API keys…**, dán **Account ID** và **API token** vào, bấm **Test** để kiểm tra rồi bấm **Save**. Không cần khởi động lại app.
   *(Cách thủ công: copy `cloudflare.example.txt` thành `cloudflare.txt`, dòng 1 là Account ID, dòng 2 là token.)*
5. Bấm **Test** thấy dòng **Token is valid ✓** là đã kết nối được. Từ câu tiếp theo, Japo dịch bằng AI (Gemma).

*(Tuỳ chọn)* **Gemini:** tạo key tại [Google AI Studio](https://aistudio.google.com/apikey), rồi dán vào ô **Gemini** trong cửa sổ **API keys**.

> ⚠️ **Không bao giờ đưa `cloudflare.txt` hay `gemini_key.txt` lên GitHub hay gửi cho người khác.** File `.gitignore` đã chặn sẵn 2 file này.

---

## 4. Cách dùng

1. Mở Japo (`run.bat`), rồi mở video tiếng Nhật và phát như bình thường.
2. Japo tự nghe âm thanh từ **loa/tai nghe mặc định** của Windows và hiện phụ đề.
3. Mỗi câu hiện theo thứ tự: **giờ + ♀/♂** → **romaji** → *tiếng Anh* → **tiếng Việt** (chữ đậm).

### Thanh công cụ
| Nút | Chức năng |
|---|---|
| ⏸ / ▶ | Tạm dừng / tiếp tục nghe |
| 🗑 | Xoá toàn bộ phụ đề đang hiện |
| **JP** / **EN** | Ẩn/hiện dòng romaji (hoặc chữ Nhật) / dòng tiếng Anh |
| ⊖ / ⊕ | Thu nhỏ / phóng to chữ |
| ☀ / ☾ | Đổi giao diện sáng / tối |
| ⚙ | Mở menu cài đặt |

### Ô bối cảnh phim
Ô ngay dưới thanh công cụ: nhập **tên phim, nội dung, tên nhân vật và quan hệ giữa họ**. AI sẽ dựa vào đó để dịch đúng tên riêng và xưng hô cho hợp. Nội dung tự lưu.

### Menu ⚙ Settings
- **Light theme:** giao diện sáng
- **Dock left / right:** đưa cửa sổ về sát mép trái/phải màn hình
- **Background opacity:** làm nền trong suốt, chữ vẫn rõ
- **Always on top:** luôn nằm trên các cửa sổ khác
- **Uncensored translation:** dịch sát nghĩa, không nói giảm nói tránh
- **Show romaji:** hiện phiên âm Latin thay cho chữ Nhật
- **Detect speaker gender:** đoán giọng nam/nữ
- **API keys…:** nhập Account ID, token Cloudflare và key Gemini
- **Save subtitles (.txt):** lưu toàn bộ phụ đề ra file
- **Open Japo folder:** mở thư mục chứa app

### Thanh trạng thái (dưới cùng)
Vạch âm lượng • đèn trạng thái (**Listening / Translating / Paused**) • số câu hoặc số câu đang chờ • GPU/CPU.

---

## 5. Mẹo và xử lý sự cố

| Vấn đề | Cách xử lý |
|---|---|
| Đeo tai nghe nhưng không nghe được | Để âm lượng **trong trình phát** 50–100%, giảm âm lượng **Windows** cho vừa tai. Japo có tự khuếch đại |
| Không hiện phụ đề | Kiểm tra vạch âm lượng có nhảy không. Nếu vừa đổi loa/tai nghe thì tắt Japo mở lại |
| Chậm, nhiều câu "queued" | Mở `japo.py`, đổi `MODEL_SIZE = "small"` |
| Lỗi GPU / cuDNN | Japo tự chuyển sang CPU. Hoặc đặt `DEVICE = "cpu"` |
| Nhạc nền to, câu dính liền nhau | Tăng `SENSITIVITY` (ví dụ `0.3`) |
| Bỏ sót câu nói nhỏ | Giảm `SENSITIVITY` (ví dụ `0.15`) |
| Muốn phụ đề tiếng Anh | Đặt `TARGET_LANG = "en"` |
| Cloudflare lỗi 401 | Token sai hoặc thiếu quyền **Workers AI**, tạo token mới theo mẫu Workers AI |
| Không hiện romaji | Chạy lại `setup.bat` (thiếu thư viện `cutlet`) |
| `No module named ...` sau khi chuyển thư mục | Luôn mở bằng `run.bat`. Nếu vẫn lỗi, xoá thư mục `venv` rồi chạy lại `setup.bat` |
| App bản exe không chạy | Xem file `japo_log.txt` trong thư mục app |

Các thông số trên nằm ở phần **SETTINGS** đầu file `japo.py`.

---

## 6. Đóng gói thành app (tuỳ chọn)

- **`build_exe.bat`:** tạo app chạy độc lập trong `dist\Japo` (khoảng 1.5–2 GB), kèm lối tắt ngoài Desktop. Máy khác dùng bản này không cần cài Python.
- **`build_installer.bat`:** gói `dist\Japo` thành **một file cài đặt** `Output\Japo-Setup.exe`. Tự cài công cụ Inno Setup nếu máy chưa có. Phải chạy `build_exe.bat` trước.

Model không nằm trong bộ cài (quá nặng), app sẽ tự tải ở lần chạy đầu. Nếu lúc build có `cloudflare.txt` trong thư mục thì bộ cài sẽ chứa luôn token của bạn, **đừng chia sẻ bộ cài đó**.

---

## 7. Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `japo.py` | Mã nguồn chính |
| `japo_icons.py` | Icon giao diện (ảnh nhúng sẵn) |
| `icon.ico`, `icon_light.ico` | Icon app (bản tối / bản sáng, tự đổi theo theme) |
| `requirements.txt` | Danh sách thư viện Python |
| `setup.bat` | Cài môi trường + thư viện |
| `run.bat` | Chạy Japo |
| `build_exe.bat` | Đóng gói thành app `.exe` |
| `build_installer.bat`, `japo.iss` | Tạo bộ cài `Japo-Setup.exe` |
| `cloudflare.example.txt` | Mẫu file cấu hình Cloudflare |
| `.gitignore` | Chặn file cá nhân/nặng khi đưa lên GitHub |

**Tự sinh ra khi chạy (không đưa lên GitHub):** `venv\`, `model\`, `dist\`, `build\`, `Output\`, `cloudflare.txt`, `gemini_key.txt`, `japo_settings.json`, `japo_log.txt`.

---

## 8. Giới hạn

- Phụ đề trễ khoảng 2–5 giây so với lời thoại.
- Nhận diện kém khi nhiều người nói chồng lên nhau, nhạc nền rất to, hoặc lời bị lẫn với tiếng thở/cảm thán.
- Đoán giọng nam/nữ dựa trên cao độ giọng, có thể sai với giọng trầm/cao bất thường.
- Máy dịch miễn phí (Microsoft/Google) có thể thay đổi hoặc giới hạn bất cứ lúc nào.

---

## Góp ý & báo lỗi

Gặp lỗi hoặc có ý tưởng, hãy mở [Issues](https://github.com/Maditor/Japo/issues) trên GitHub. Nhớ kèm nội dung cửa sổ cmd hoặc file `japo_log.txt` (**xoá token trước khi gửi**).

Made by [Maditor](https://github.com/Maditor).
