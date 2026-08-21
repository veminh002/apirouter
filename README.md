# ApiRouter v3

Cổng (gateway) AI tương thích OpenAI, có provider registry, alias model, routing fallback, streaming SSE thật, circuit breaker, health check và metrics.

## Kiến trúc

```text
Client
  -> FastAPI /v1/chat/completions
  -> Auth + rate limit
  -> RoutingPolicy / ModelAlias
  -> ProviderRouter
       -> Tavily search (nếu TAVILY_SEARCH_MODE != off, chèn context 1 lần)
       -> CircuitBreaker
       -> retry / Retry-After
       -> ProviderRegistry
            -> ChatGPT-Web (PRIMARY, xác thực bằng refresh-token, SSE thật)
            -> NVIDIA NIM (fallback, ngay sau ChatGPT)
            -> TokenRouter (fallback)
            -> Groq (fallback)
            -> OpenRouter (fallback)
  -> Phản hồi / SSE tương thích OpenAI
```

## Các endpoint chính

- `GET /health` - trạng thái health của provider + trạng thái circuit breaker
- `GET /metrics` - metrics dạng text Prometheus
- `GET /metrics/json` - metrics dạng JSON
- `GET /v1/models` - danh sách alias model logic đã cấu hình
- `POST /v1/chat/completions` - completion tương thích OpenAI, hỗ trợ streaming SSE thật

## Alias model

Client có thể tiếp tục gửi tên logic như `gpt-4o-mini`, router sẽ tự chọn model thật của provider tương ứng.

Có thể override routing bằng biến môi trường, ví dụ:

```env
ALIAS_GPT_4O_MINI=groq:openai/gpt-oss-20b,openrouter:google/gemma-4-26b-a4b-it:free
```

Muốn route thẳng, dùng cú pháp `provider:model`, ví dụ `openrouter:google/gemini-2.5-flash`.

## Circuit breaker

Một provider sẽ mở circuit sau khi có `CIRCUIT_FAILURE_THRESHOLD` lần lỗi có thể retry. Sau `CIRCUIT_RECOVERY_SECONDS`, hệ thống cho phép 1 request half-open để thử dò lại khả năng phục hồi.

## Streaming

ChatGPT-Web là provider streaming chính. Nó dùng OAuth refresh token để lấy access token, gọi endpoint conversation của ChatGPT Web bằng SSE, và forward các delta gia tăng. Groq và OpenRouter vẫn là fallback hỗ trợ streaming.

Router có thể chuyển provider trong suốt trước khi chunk stream đầu tiên được gửi đi. Sau khi đã có chunk tới client, việc đổi provider sẽ làm hỏng luồng hội thoại, nên router sẽ phát lỗi stream thay vì âm thầm chuyển đổi.

## Chạy local

```bash
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 10000
```

## Thứ tự routing

ChatGPT Web là provider chính. Với mỗi alias model logic, thứ tự routing là: ChatGPT trước, rồi tới NVIDIA NIM, rồi tới TokenRouter, rồi tới Groq, rồi tới OpenRouter.

## NVIDIA NIM

Tầng fallback ngay sau ChatGPT, dùng endpoint OpenAI-compatible miễn phí của
`build.nvidia.com` (`https://integrate.api.nvidia.com/v1`). Đặt `NVIDIA_API_KEY`
(lấy free tại build.nvidia.com) và `ENABLE_NVIDIA=true` để bật. Model dùng cho
cả 4 alias là `nvidia/nemotron-3-ultra-550b-a55b` — Free Endpoint trên
build.nvidia.com tại thời điểm thêm (20/08/2026). `mistralai/mistral-nemotron`
ban đầu dùng cho alias `gpt-4o-mini`/`gpt-3.5-turbo` nhưng đã đổi sang
`nemotron-3-ultra`: model nhỏ đó tiếng Việt kém, hay hallucinate persona và
dữ liệu bịa cho cả câu hỏi đơn giản không cần search.
`nemotron-3-ultra` là reasoning-hybrid model và mặc định trả về chain-of-thought
thô lẫn vào `content`; provider tự tắt `enable_thinking` trừ khi caller tự set.

## Xác thực ChatGPT

Đặt `CHATGPT_REFRESH_TOKEN` trong `.env`. Ứng dụng refresh access token qua cùng luồng Auth0 refresh-token mà bản triển khai ApiRouter gốc sử dụng, và cache access token cho tới gần lúc hết hạn. Tuyệt đối không commit refresh token vào source control.

## Streaming

Khi `stream=true`, SSE của ChatGPT Web được đọc tuần tự và chuyển đổi thành các chunk SSE tương thích OpenAI. Vì endpoint ChatGPT Web gửi text dạng tích lũy, provider chỉ phát ra phần delta text mới được thêm vào. Nếu một stream đã phát dữ liệu ra rồi, router sẽ không đổi provider giữa chừng.

## Độ bền của token ChatGPT

ChatGPT Web vẫn là provider chính. Router cache access token, chấp nhận và lưu lại refresh token đã rotate khi được trả về, và có thể chạy một tác vụ nền keep-alive để refresh định kỳ. Đặt `CHATGPT_TOKEN_STATE_FILE` trỏ tới một đường dẫn lưu trữ bền vững để token đã rotate không bị mất khi restart, và dùng `CHATGPT_KEEPALIVE_HOURS` để điều chỉnh tần suất refresh (mặc định 6 giờ). Keep-alive có thể reset thời gian sống idle của phía cấp token khi được cho phép, nhưng không client nào có thể kéo dài thời gian sống tối đa tuyệt đối do phía cấp token quy định.

## Tìm kiếm web: Tavily tập trung, tắt search riêng ở từng provider

Tất cả tìm kiếm đi qua **Tavily** một lần duy nhất, ở `ProviderRouter`, trước
khi request được thử qua bất kỳ provider nào (ChatGPT, NVIDIA, TokenRouter,
Groq, OpenRouter). Kết quả search được chèn thành một message `role: system`
ngay trước câu hỏi cuối của user — nên **mọi provider trong chuỗi fallback
đều thấy cùng một context tìm kiếm**, kể cả khi router phải fallback từ
ChatGPT sang Groq giữa chừng.

Cấu hình:

```env
TAVILY_API_KEY=tvly-...
TAVILY_SEARCH_MODE=auto   # off | auto | always
TAVILY_MAX_RESULTS=5
```

- `off`: không gọi Tavily, không chèn gì (mặc định nếu không set `TAVILY_API_KEY`).
- `auto`: chỉ gọi Tavily khi câu hỏi có vẻ cần thông tin mới (dùng chung
  `detect_realtime` với cơ chế cũ của ChatGPT).
- `always`: gọi Tavily cho mọi request.

Nếu Tavily lỗi (mạng, hết quota, API key sai...), router log warning và
**tiếp tục xử lý bình thường không có context search**, không làm fail request.

**Search native của từng provider đã tắt hết** (`CHATGPT_WEB_SEARCH_MODE=off`,
`GROQ_WEB_SEARCH_MODE=off`, `NVIDIA_/OPENROUTER_/TOKENROUTER_WEB_SEARCH_MODE=off`)
để tránh trùng lặp — chỉ Tavily quyết định khi nào cần search. Muốn quay lại
cơ chế cũ (ChatGPT/Groq tự search riêng, không qua Tavily) thì set các biến
đó về `auto`/`always` như trước.

## Vision (đọc ảnh)

Client gửi ảnh theo đúng format chuẩn OpenAI chat-completions:

```json
{"role": "user", "content": [
  {"type": "text", "text": "Ảnh này có gì?"},
  {"type": "image_url", "image_url": {"url": "https://... hoặc data:image/png;base64,..."}}
]}
```

- **Groq / OpenRouter / TokenRouter**: payload được forward nguyên vẹn tới
  upstream (`req.model_dump`), nên khối `image_url` đã tự động đi qua —
  không cần sửa gì, miễn là **model đích có hỗ trợ vision** (vd. trên Groq:
  `meta-llama/llama-4-scout-17b-16e-instruct`, `meta-llama/llama-4-maverick-17b-128e-instruct`;
  trên OpenRouter: bất kỳ model đa phương thức nào như `openai/gpt-4o`,
  `google/gemini-2.5-flash`, v.v.). Nếu alias của bạn trỏ tới model text-only,
  ảnh sẽ bị model đó bỏ qua hoặc lỗi — bạn cần trỏ alias sang model vision.
- **ChatGPT**: trước đây bị bỏ ảnh hoàn toàn (code cũ gộp cả hội thoại thành
  một chuỗi text và loại bỏ mọi khối không phải `text`). Đã sửa: mỗi message
  giờ được convert thành một Responses `input` item riêng, giữ nguyên
  `input_image` cho ảnh, nên ChatGPT giờ nhìn thấy ảnh đúng như gửi lên.
