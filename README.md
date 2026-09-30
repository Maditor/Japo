# Japo

**Tác giả:** [Maditor](https://github.com/Maditor) · **Repo:** [github.com/Maditor/Japo](https://github.com/Maditor/Japo)

Phụ đề **thời gian thực** cho video tiếng **Nhật, Hàn, Trung và Anh** trên Windows.
Japo nghe âm thanh đang phát trên máy (YouTube, trình phát phim, trình duyệt…), nhận diện lời thoại bằng **Whisper**, dịch sang **tiếng Việt** (hoặc 11 ngôn ngữ khác) và hiện từng câu trong một thanh dọc gọn gàng bên cạnh màn hình.

- Không cần file phụ đề, không cần tải video về
- Nghe được **tiếng Nhật, Hàn, Trung, Anh**, hoặc để Japo **tự nhận ngôn ngữ**
- Dịch bằng AI (Cloudflare, Groq, OpenRouter, Gemini…) có ngữ cảnh nên xưng hô tự nhiên. Hết lượt thì **tự chuyển dịch vụ**
- Hiện kèm **phiên âm** (romaji / pinyin / tiếng Hàn Latin) và bản **tiếng Anh**
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

## 3. Cấu hình dịch (Translation & API keys)

Mở Japo → **⚙ → Translation & API keys…**. Cửa sổ gồm 2 phần:

- **Bên trái – danh sách dịch vụ theo thứ tự ưu tiên.** Tick ô vuông để bật/tắt, dùng ▲ ▼ để đổi thứ tự. Chấm **xanh** là đã cài đủ, chấm **xám** là còn thiếu key.
- **Bên phải – cài đặt của dịch vụ đang chọn:** key, model, link lấy key và nút **Test translation** để dịch thử một câu.

Japo dùng dịch vụ **đầu tiên đang bật và đã cài đủ**. Nếu dịch vụ đó lỗi hoặc hết lượt, Japo tự chuyển xuống dịch vụ tiếp theo.

| Dịch vụ | Cần gì | Ghi chú |
|---|---|---|
| **Cloudflare Workers AI** | Account ID + API token | Dịch hay nhất (Gemma 4), có lượt miễn phí mỗi ngày |
| **Groq** | API key + model | Rất nhanh, gói miễn phí có giới hạn tốc độ |
| **OpenRouter** | API key + model | Nhiều model, model đuôi `:free` miễn phí |
| **Custom** | Base URL + model | Mọi dịch vụ kiểu OpenAI, kể cả AI chạy trên máy (LM Studio, Ollama) |
| **Google Gemini** | API key | Gói miễn phí có giới hạn theo ngày |
| **Microsoft Translator** | Không cần | Miễn phí, nhanh, nhưng dịch từng câu, không có ngữ cảnh |
| **Google Translate** | Không cần | Như trên |

**Lấy key Cloudflare:** vào [trang API Tokens](https://dash.cloudflare.com/profile/api-tokens) → **Create Token** → mẫu **Workers AI**. Chú ý là **API Token**, không phải *Global API Key*. Account ID là dãy 32 ký tự trên thanh địa chỉ ngay sau `dash.cloudflare.com/`.

**Chọn model cho Groq / OpenRouter / Custom:** dán API key rồi bấm **Browse…** để xem danh sách model và chọn (OpenRouter có ô lọc *Free models only*).

**Khi hết lượt:** Cloudflare báo hết lượt miễn phí trong ngày thì Japo tạm bỏ qua nó đến lúc lượt được làm mới (7 giờ sáng giờ Việt Nam). Các dịch vụ khác bị giới hạn thì nghỉ vài phút. Nếu tất cả đều lỗi, Japo vẫn luôn còn máy dịch làm phương án cuối. Thanh dưới cùng hiện tên dịch vụ đang dùng.

> ⚠️ Key được lưu trong file **`translators.json`** cạnh Japo. **Không bao giờ đưa file này lên GitHub hay gửi cho người khác.** `.gitignore` đã chặn sẵn. (Bản cũ dùng `cloudflare.txt`, `gemini_key.txt`, `ai_custom.json`: Japo tự nhập các file này ở lần mở đầu tiên.)

---

## 4. Cách dùng

1. Mở Japo (`run.bat`), rồi mở video (tiếng Nhật, Hàn, Trung hoặc Anh) và phát như bình thường.
2. Japo tự nghe âm thanh từ **loa/tai nghe mặc định** của Windows và hiện phụ đề.
3. Mỗi câu hiện theo thứ tự: **giờ + ♀/♂** → **phiên âm** → *tiếng Anh* → **bản dịch** (chữ đậm).

### Chọn ngôn ngữ
Vào **⚙ Settings**:
- **Audio language:** ngôn ngữ của video: *Auto detect*, Japanese, Korean, Chinese, English. Biết trước video tiếng gì thì chọn thẳng sẽ chính xác hơn *Auto*.
- **Translate to:** ngôn ngữ phụ đề: Vietnamese, English, Japanese, Korean, Chinese (giản thể/phồn thể), Thai, Indonesian, French, Spanish, German, Russian.

Lựa chọn được tự lưu. Thanh dưới cùng hiện cặp ngôn ngữ đang dùng, ví dụ `KO→VI`.

| Ngôn ngữ nghe | Phiên âm hiển thị |
|---|---|
| Tiếng Nhật | romaji (*arigatou*) |
| Tiếng Trung | pinyin có dấu thanh (*xiè xiè*) |
| Tiếng Hàn | chữ Latin (*gomawoyo*) |
| Tiếng Anh | không cần |

### Thanh công cụ
| Nút | Chức năng |
|---|---|
| ⏸ / ▶ | Tạm dừng / tiếp tục nghe |
| 🗑 | Xoá toàn bộ phụ đề đang hiện |
| **JA/KO/ZH/EN** / **EN** | Nút đầu đổi theo ngôn ngữ đang nghe: ẩn/hiện câu gốc (hoặc phiên âm). Nút sau: ẩn/hiện dòng tiếng Anh |
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
- **Audio language / Translate to:** chọn ngôn ngữ nghe và ngôn ngữ dịch
- **Show romanization:** hiện phiên âm Latin thay cho chữ gốc
- **Detect speaker gender:** đoán giọng nam/nữ
- **Translation & API keys…:** chọn dịch vụ dịch, thứ tự ưu tiên và nhập key
- **Save subtitles (.txt):** lưu toàn bộ phụ đề ra file
- **Open Japo folder:** mở thư mục chứa app

### Thanh trạng thái (dưới cùng)
Vạch âm lượng • đèn trạng thái (**Listening / Translating / Paused**) • số câu (`3⏳` nghĩa là còn 3 câu đang chờ) • cặp ngôn ngữ • dịch vụ vừa dịch. Chữ **CPU** chỉ hiện khi Japo không dùng được GPU.

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
| Muốn phụ đề tiếng Anh | ⚙ Settings → **Translate to** → English |
| Auto detect nhận nhầm ngôn ngữ | Chọn thẳng ngôn ngữ trong **Audio language** |
| Cloudflare lỗi 401 | Token sai hoặc thiếu quyền **Workers AI**, tạo token mới theo mẫu Workers AI |
| Không hiện phiên âm | Chạy lại `setup.bat` (thiếu `cutlet`, `pypinyin` hoặc `korean-romanizer`). Bản exe thì build lại sau khi cài |
| `translation failed – ModuleNotFoundError` | Thiếu thư viện `requests`: chạy lại `setup.bat` |
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
| `.gitignore` | Chặn file cá nhân/nặng khi đưa lên GitHub |

**Tự sinh ra khi chạy (không đưa lên GitHub):** `venv\`, `model\`, `dist\`, `build\`, `Output\`, `translators.json`, `japo_settings.json`, `japo_log.txt`.

---

## 8. Giới hạn

- Phụ đề trễ khoảng 2–5 giây so với lời thoại.
- Nhận diện kém khi nhiều người nói chồng lên nhau, nhạc nền rất to, hoặc lời bị lẫn với tiếng thở/cảm thán.
- Đoán giọng nam/nữ dựa trên cao độ giọng, có thể sai với giọng trầm/cao bất thường.
- Bộ lọc tiếng thở/cảm thán và câu "ảo" (Whisper tự bịa khi im lặng) được tối ưu cho tiếng Nhật; với ngôn ngữ khác thỉnh thoảng có thể lọt câu kiểu *"Thanks for watching"*.
- Máy dịch miễn phí (Microsoft/Google) có thể thay đổi hoặc giới hạn bất cứ lúc nào.

---

## Góp ý & báo lỗi

Gặp lỗi hoặc có ý tưởng, hãy mở [Issues](https://github.com/Maditor/Japo/issues) trên GitHub. Nhớ kèm nội dung cửa sổ cmd hoặc file `japo_log.txt` (**xoá token trước khi gửi**).

Made by [Maditor](https://github.com/Maditor).
