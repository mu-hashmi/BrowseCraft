package dev.browsecraft.mod;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.WebSocket;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.Objects;

public final class BackendClient implements BuildBackend {
    private static final long RECONNECT_BASE_DELAY_MS = 500;
    private static final long RECONNECT_MAX_DELAY_MS = 30_000;

    private final BackendEndpoints endpoints;
    private final String mcVersion;
    private final HttpClient httpClient;
    private final Gson gson = new Gson();
    private final ScheduledExecutorService reconnectExecutor;
    private final AtomicInteger reconnectAttempts = new AtomicInteger();
    private final AtomicLong connectionGeneration = new AtomicLong();
    private final AtomicReference<String> activeJobId = new AtomicReference<>();
    private final ToolRequestHandler toolRequestHandler;

    private volatile BuildBackendListener listener;
    private volatile WebSocket webSocket;
    private volatile boolean closed;
    private volatile boolean seenSuccessfulConnection;

    public BackendClient(BackendEndpoints endpoints, String mcVersion) {
        this(endpoints, mcVersion, (tool, params) -> CompletableFuture.failedFuture(
                new UnsupportedOperationException("Tool request handler is not configured")
        ));
    }

    public BackendClient(BackendEndpoints endpoints, String mcVersion, ToolRequestHandler toolRequestHandler) {
        this.endpoints = endpoints;
        this.mcVersion = mcVersion;
        this.toolRequestHandler = Objects.requireNonNull(toolRequestHandler, "toolRequestHandler");
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .version(HttpClient.Version.HTTP_1_1)
                .build();
        this.reconnectExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "browsecraft-backend-ws");
            thread.setDaemon(true);
            return thread;
        });
    }

    @Override
    public void connect(BuildBackendListener listener) {
        this.listener = listener;
        this.closed = false;
        long generation = this.connectionGeneration.incrementAndGet();
        openWebSocket(generation);
    }

    @Override
    public String submitBuildQuery(String query, String clientId) throws IOException, InterruptedException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("query", query);
        requestBody.addProperty("mc_version", mcVersion);
        requestBody.addProperty("client_id", clientId);
        return submitJobRequest("/v1/jobs", requestBody);
    }

    @Override
    public String submitImaginePrompt(String prompt, String clientId) throws IOException, InterruptedException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("prompt", prompt);
        requestBody.addProperty("client_id", clientId);
        return submitJobRequest("/v1/imagine", requestBody);
    }

    @Override
    public String submitImagineModifyPrompt(String prompt, String clientId) throws IOException, InterruptedException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("prompt", prompt);
        requestBody.addProperty("client_id", clientId);
        return submitJobRequest("/v1/imagine/modify", requestBody);
    }

    @Override
    public void submitChatMessage(
            String message,
            String clientId,
            String worldId,
            String sessionId
    ) throws IOException, InterruptedException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("client_id", clientId);
        requestBody.addProperty("message", message);
        requestBody.addProperty("world_id", worldId);
        if (sessionId != null && !sessionId.isBlank()) {
            requestBody.addProperty("session_id", sessionId);
        }
        postJson("/v1/chat", requestBody);
    }

    @Override
    public String createSession(String clientId, String worldId) throws IOException, InterruptedException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("client_id", clientId);
        requestBody.addProperty("world_id", worldId);
        HttpResponse<String> response = postJson("/v1/session/new", requestBody);
        JsonObject payload = JsonParser.parseString(response.body()).getAsJsonObject();
        return requiredString(payload, "session_id");
    }

    @Override
    public List<String> listSessions(String clientId, String worldId) throws IOException, InterruptedException {
        HttpResponse<String> response = getJson(
                "/v1/session/list",
                Map.of(
                        "client_id", clientId,
                        "world_id", worldId
                )
        );
        JsonElement parsed = JsonParser.parseString(response.body());
        JsonArray sessions = extractSessionsArray(parsed);

        List<String> sessionIds = new ArrayList<>(sessions.size());
        for (JsonElement entry : sessions) {
            if (entry.isJsonPrimitive() && entry.getAsJsonPrimitive().isString()) {
                sessionIds.add(entry.getAsString());
                continue;
            }
            if (entry.isJsonObject()) {
                sessionIds.add(requiredString(entry.getAsJsonObject(), "session_id"));
                continue;
            }
            throw new IllegalArgumentException("Session list contains unsupported entry: " + entry);
        }
        return List.copyOf(sessionIds);
    }

    @Override
    public void switchSession(String clientId, String worldId, String sessionId) throws IOException, InterruptedException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("client_id", clientId);
        requestBody.addProperty("world_id", worldId);
        requestBody.addProperty("session_id", sessionId);
        postJson("/v1/session/switch", requestBody);
    }

    private String submitJobRequest(String endpoint, JsonObject requestBody) throws IOException, InterruptedException {
        HttpResponse<String> response = postJson(endpoint, requestBody);
        JsonObject payload = JsonParser.parseString(response.body()).getAsJsonObject();
        String jobId = payload.get("job_id").getAsString();
        activeJobId.set(jobId);
        return jobId;
    }

    private HttpResponse<String> postJson(String endpoint, JsonObject requestBody) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(endpoints.baseUrl() + endpoint))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(requestBody)))
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("Backend returned status " + response.statusCode() + ": " + response.body());
        }
        return response;
    }

    private HttpResponse<String> getJson(String endpoint, Map<String, String> queryParams) throws IOException, InterruptedException {
        StringBuilder uriBuilder = new StringBuilder(endpoints.baseUrl()).append(endpoint);
        if (!queryParams.isEmpty()) {
            uriBuilder.append("?");
            boolean first = true;
            for (Map.Entry<String, String> entry : queryParams.entrySet()) {
                if (!first) {
                    uriBuilder.append("&");
                }
                uriBuilder.append(URLEncoder.encode(entry.getKey(), StandardCharsets.UTF_8));
                uriBuilder.append("=");
                uriBuilder.append(URLEncoder.encode(entry.getValue(), StandardCharsets.UTF_8));
                first = false;
            }
        }

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(uriBuilder.toString()))
                .GET()
                .build();
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new IOException("Backend returned status " + response.statusCode() + ": " + response.body());
        }
        return response;
    }

    private JsonArray extractSessionsArray(JsonElement parsed) {
        if (parsed.isJsonArray()) {
            return parsed.getAsJsonArray();
        }
        if (!parsed.isJsonObject()) {
            throw new IllegalArgumentException("Expected JSON object or array for sessions response");
        }

        JsonObject object = parsed.getAsJsonObject();
        JsonElement sessions = object.get("sessions");
        if (sessions == null || !sessions.isJsonArray()) {
            throw new IllegalArgumentException("Expected sessions array in response");
        }
        return sessions.getAsJsonArray();
    }

    @Override
    public void close() {
        closed = true;
        WebSocket current = this.webSocket;
        if (current != null) {
            current.sendClose(WebSocket.NORMAL_CLOSURE, "closing");
        }
        reconnectExecutor.shutdownNow();
    }

    private void openWebSocket(long generation) {
        httpClient.newWebSocketBuilder()
                .buildAsync(URI.create(endpoints.wsUrl()), new SocketListener(generation))
                .whenComplete((socket, error) -> {
                    if (error != null) {
                        handleConnectionFailure(generation, error);
                        return;
                    }
                    this.webSocket = socket;
                    this.reconnectAttempts.set(0);
                    this.seenSuccessfulConnection = true;
                });
    }

    private void handleConnectionFailure(long generation, Throwable error) {
        if (closed || generation != connectionGeneration.get()) {
            return;
        }

        long delay = backoffDelayMillis(reconnectAttempts.getAndIncrement());
        BuildBackendListener currentListener = listener;
        boolean hasActiveJob = activeJobId.get() != null;
        if (currentListener != null && (seenSuccessfulConnection || hasActiveJob)) {
            currentListener.onError("", "WS_DISCONNECTED", error.getMessage());
        }

        reconnectExecutor.schedule(
                () -> {
                    if (closed || generation != connectionGeneration.get()) {
                        return;
                    }
                    openWebSocket(generation);
                },
                delay,
                TimeUnit.MILLISECONDS
        );
    }

    private long backoffDelayMillis(int attempts) {
        long exponent = 1L << Math.min(attempts, 16);
        return Math.min(RECONNECT_MAX_DELAY_MS, RECONNECT_BASE_DELAY_MS * exponent);
    }

    private void handleEventMessage(JsonObject event) {
        BuildBackendListener currentListener = listener;
        if (currentListener == null) {
            return;
        }

        String type = event.get("type").getAsString();
        String jobId = event.get("job_id").getAsString();
        JsonObject payload = event.getAsJsonObject("payload");

        switch (type) {
            case "job.status" -> {
                String stage = payload.get("stage").getAsString();
                String text = payload.get("message").getAsString();
                currentListener.onStatus(jobId, stage, text);
            }
            case "job.ready" -> {
                JsonObject source = payload.getAsJsonObject("source");
                String sourceType = source.get("type").getAsString();
                String sourceUrl = source.get("url").getAsString();
                double confidence = payload.get("confidence").getAsDouble();
                BuildPlan plan = parsePlan(payload.getAsJsonObject("plan"));
                activeJobId.compareAndSet(jobId, null);
                currentListener.onReady(jobId, sourceType, sourceUrl, confidence, plan);
            }
            case "job.error" -> {
                String code = payload.get("code").getAsString();
                String text = payload.get("message").getAsString();
                if (code.startsWith("WS_") && activeJobId.get() == null) {
                    return;
                }
                activeJobId.compareAndSet(jobId, null);
                currentListener.onError(jobId, code, text);
            }
            default -> {
            }
        }
    }

    private void handleIncomingMessage(WebSocket socket, String message) {
        JsonObject envelope = JsonParser.parseString(message).getAsJsonObject();
        String type = requiredString(envelope, "type");
        if ("chat.response".equals(type)) {
            handleChatResponse(envelope);
            return;
        }
        if ("chat.delta".equals(type)) {
            handleChatDelta(envelope);
            return;
        }
        if ("tool.request".equals(type)) {
            handleToolRequest(socket, envelope);
            return;
        }
        handleEventMessage(envelope);
    }

    private void handleChatResponse(JsonObject envelope) {
        BuildBackendListener currentListener = listener;
        if (currentListener == null) {
            return;
        }

        JsonObject payload = requiredObject(envelope, "payload");
        String message = requiredString(payload, "message");
        currentListener.onStatus("", "chat", message);
    }

    private void handleChatDelta(JsonObject envelope) {
        BuildBackendListener currentListener = listener;
        if (currentListener == null) {
            return;
        }

        JsonObject payload = requiredObject(envelope, "payload");
        String partial = requiredString(payload, "partial");
        currentListener.onStatus("", "chat", partial);
    }

    private void handleToolRequest(WebSocket socket, JsonObject envelope) {
        String requestId = null;
        try {
            requestId = requiredString(envelope, "request_id");
            String tool = requiredString(envelope, "tool");
            JsonObject params = requiredObject(envelope, "params");
            CompletableFuture<JsonObject> dispatch = toolRequestHandler.handle(tool, params);
            String responseRequestId = requestId;
            dispatch.whenComplete((result, error) -> {
                if (error != null) {
                    sendToolError(socket, responseRequestId, rootCause(error));
                    return;
                }
                if (result == null) {
                    sendToolError(socket, responseRequestId, new IllegalStateException("Tool handler returned null result"));
                    return;
                }
                sendToolResult(socket, responseRequestId, result);
            });
        } catch (RuntimeException error) {
            if (requestId != null && !requestId.isEmpty()) {
                sendToolError(socket, requestId, error);
            }
        }
    }

    private void sendToolResult(WebSocket socket, String requestId, JsonObject result) {
        JsonObject payload = new JsonObject();
        payload.addProperty("type", "tool.response");
        payload.addProperty("request_id", requestId);
        payload.add("result", result);
        socket.sendText(gson.toJson(payload), true);
    }

    private void sendToolError(WebSocket socket, String requestId, Throwable error) {
        JsonObject payload = new JsonObject();
        payload.addProperty("type", "tool.response");
        payload.addProperty("request_id", requestId);
        String message = error.getMessage();
        if (message == null) {
            message = error.toString();
        }
        payload.addProperty("error", message);
        socket.sendText(gson.toJson(payload), true);
    }

    private String requiredString(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonPrimitive() || !element.getAsJsonPrimitive().isString()) {
            throw new IllegalArgumentException("Expected string field: " + key);
        }
        return element.getAsString();
    }

    private JsonObject requiredObject(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonObject()) {
            throw new IllegalArgumentException("Expected object field: " + key);
        }
        return element.getAsJsonObject();
    }

    private Throwable rootCause(Throwable error) {
        if (error.getCause() == null) {
            return error;
        }
        return error.getCause();
    }

    private BuildPlan parsePlan(JsonObject planObject) {
        int totalBlocks = planObject.get("total_blocks").getAsInt();
        List<BuildPlacement> placements = new ArrayList<>();

        for (JsonElement placementElement : planObject.getAsJsonArray("placements")) {
            JsonObject placementObject = placementElement.getAsJsonObject();
            JsonObject blockStateObject = placementObject.getAsJsonObject("block_state");
            Map<String, String> blockState = new HashMap<>();
            if (blockStateObject != null) {
                for (Map.Entry<String, JsonElement> entry : blockStateObject.entrySet()) {
                    blockState.put(entry.getKey(), entry.getValue().getAsString());
                }
            }

            placements.add(new BuildPlacement(
                    placementObject.get("dx").getAsInt(),
                    placementObject.get("dy").getAsInt(),
                    placementObject.get("dz").getAsInt(),
                    placementObject.get("block_id").getAsString(),
                    Map.copyOf(blockState)
            ));
        }

        return new BuildPlan(totalBlocks, List.copyOf(placements));
    }

    private final class SocketListener implements WebSocket.Listener {
        private final long generation;
        private final StringBuilder pendingText = new StringBuilder();

        private SocketListener(long generation) {
            this.generation = generation;
        }

        @Override
        public void onOpen(WebSocket webSocket) {
            WebSocket.Listener.super.onOpen(webSocket);
            webSocket.request(1);
        }

        @Override
        public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
            pendingText.append(data);
            if (last) {
                String message = pendingText.toString();
                pendingText.setLength(0);
                handleIncomingMessage(webSocket, message);
            }
            webSocket.request(1);
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
            handleConnectionFailure(generation, new IOException("WebSocket closed: " + statusCode + " " + reason));
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public void onError(WebSocket webSocket, Throwable error) {
            handleConnectionFailure(generation, error);
        }
    }

    @FunctionalInterface
    public interface ToolRequestHandler {
        CompletableFuture<JsonObject> handle(String tool, JsonObject params);
    }
}
