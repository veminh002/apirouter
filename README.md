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
ALIAS_GPT_4O_MINI=groq:llama-3.1-8b-instant,openrouter:google/gemini-2.0-flash-001
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
uvicorn app:app --host 0.0.0.0 --port 10000
```

## Thứ tự routing

ChatGPT Web là provider chính. Với mỗi alias model logic, thứ tự routing là: ChatGPT trước, rồi tới Groq, rồi tới OpenRouter.

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

## ChatGPT Web / Codex authentication

For ChatGPT subscription routing, use a current ChatGPT/Codex OAuth access token and the
associated `account_id`. Do not use the encrypted `sessionToken` from
`/api/auth/session`. v7 can extract `chatgpt_account_id` from the JWT automatically or
accept it explicitly through `CHATGPT_ACCOUNT_ID`.

The ChatGPT provider targets the ChatGPT/Codex Responses backend:
`https://chatgpt.com/backend-api/codex/responses`.

If the backend returns 401, v7 includes non-secret authentication diagnostics in the
provider error. Secret token values are never logged.
