# ApiRouter v3

Cổng (gateway) AI tương thích OpenAI, có provider registry, alias model, routing fallback, streaming SSE thật, circuit breaker, health check và metrics.

## Kiến trúc

```text
Client
  -> FastAPI /v1/chat/completions
  -> Auth + rate limit
  -> RoutingPolicy / ModelAlias
  -> ProviderRouter
       -> CircuitBreaker
       -> retry / Retry-After
       -> ProviderRegistry
            -> ChatGPT-Web (PRIMARY, xác thực bằng refresh-token, SSE thật)
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

ChatGPT Web là provider chính. Với mỗi alias model logic, thứ tự routing là: ChatGPT trước, rồi tới TokenRouter, rồi tới Groq, rồi tới OpenRouter.

## Xác thực ChatGPT

Đặt `CHATGPT_REFRESH_TOKEN` trong `.env`. Ứng dụng refresh access token qua cùng luồng Auth0 refresh-token mà bản triển khai ApiRouter gốc sử dụng, và cache access token cho tới gần lúc hết hạn. Tuyệt đối không commit refresh token vào source control.

## Streaming

Khi `stream=true`, SSE của ChatGPT Web được đọc tuần tự và chuyển đổi thành các chunk SSE tương thích OpenAI. Vì endpoint ChatGPT Web gửi text dạng tích lũy, provider chỉ phát ra phần delta text mới được thêm vào. Nếu một stream đã phát dữ liệu ra rồi, router sẽ không đổi provider giữa chừng.

## Độ bền của token ChatGPT

ChatGPT Web vẫn là provider chính. Router cache access token, chấp nhận và lưu lại refresh token đã rotate khi được trả về, và có thể chạy một tác vụ nền keep-alive để refresh định kỳ. Đặt `CHATGPT_TOKEN_STATE_FILE` trỏ tới một đường dẫn lưu trữ bền vững để token đã rotate không bị mất khi restart, và dùng `CHATGPT_KEEPALIVE_HOURS` để điều chỉnh tần suất refresh (mặc định 6 giờ). Keep-alive có thể reset thời gian sống idle của phía cấp token khi được cho phép, nhưng không client nào có thể kéo dài thời gian sống tối đa tuyệt đối do phía cấp token quy định.

## Tìm kiếm/realtime gốc của ChatGPT Web (100% miễn phí)

ApiRouter không gọi Tavily, Brave, Bing, Google Search, hay bất kỳ API tìm kiếm ngoài nào khác.
Khi `CHATGPT_WEB_SEARCH_MODE=auto`, các request có vẻ cần thông tin thời gian thực sẽ được thêm một gợi ý nhỏ
trong request gửi tới ChatGPT Web, yêu cầu assistant gốc của ChatGPT Web dùng khả năng duyệt web/tìm kiếm riêng của nó nếu có sẵn. Bản thân router không bao giờ tự fetch trang web.

Đây chủ động là một cầu nối "best-effort" tới backend riêng tư của ChatGPT Web. Nó **không** đảm bảo
mọi request đều kích hoạt được tìm kiếm gốc, vì quyết định đó do chính ChatGPT Web kiểm soát.
Không có bất kỳ phụ thuộc tìm kiếm trả phí hay bên ngoài nào được thêm vào.

## Tìm kiếm ở các tầng còn lại (Groq, OpenRouter, TokenRouter)

Cùng cơ chế `detect_realtime` dùng cho ChatGPT được tái sử dụng cho ba provider
OpenAI-compatible còn lại, mỗi provider bật `*_WEB_SEARCH_MODE=off|auto|always`
độc lập (`auto` = chỉ bật khi câu hỏi có vẻ cần thông tin mới):

- **Groq**: không có tool tìm kiếm riêng gọi qua `tools`; thay vào đó khi cần
  tìm kiếm, request được chuyển sang `GROQ_WEB_SEARCH_MODEL` (mặc định
  `groq/compound-mini`), một model "compound" của Groq tự quyết định gọi
  web search server-side. Model gốc trong alias chỉ dùng khi không cần tìm kiếm.
  **Miễn phí** — tính theo token bình thường của Groq, không phụ phí tìm kiếm.
- **OpenRouter**: có thể thêm tool `{"type": "openrouter:web_search"}` vào
  request khi cần, nhưng **tool này tính phí theo mỗi lượt tìm kiếm** (khác
  ChatGPT/Groq). Vì vậy **mặc định TẮT** (`OPENROUTER_WEB_SEARCH_MODE=off`)
  để cả router luôn free theo mặc định. Tự chịu trách nhiệm nếu bật `auto`/`always`.
- **TokenRouter**: mặc định `off`. TokenRouter chỉ là một cổng OpenAI-compatible
  tới nhiều model khác nhau; khả năng tìm kiếm phụ thuộc vào model cụ thể phía
  sau nó có hỗ trợ hay không. Đặt `TOKENROUTER_WEB_SEARCH_MODE=auto` và
  `TOKENROUTER_WEB_SEARCH_MODEL=<model có search>` nếu biết TokenRouter của bạn
  có sẵn model như vậy.

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
