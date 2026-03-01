package dev.browsecraft.mod;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.function.Supplier;

public final class BuildCommandController implements BuildBackendListener {
    private final String clientId;
    private final BuildBackend backend;
    private final OverlayState overlayState;
    private final Executor mainExecutor;
    private final Executor workerExecutor;
    private final Consumer<String> statusSink;
    private final Supplier<String> worldIdSupplier;
    private String activeSessionId;

    public BuildCommandController(
            String clientId,
            BuildBackend backend,
            OverlayState overlayState,
            Executor mainExecutor,
            Executor workerExecutor,
            Consumer<String> statusSink,
            Supplier<String> worldIdSupplier
    ) {
        this.clientId = clientId;
        this.backend = backend;
        this.overlayState = overlayState;
        this.mainExecutor = mainExecutor;
        this.workerExecutor = workerExecutor;
        this.statusSink = statusSink;
        this.worldIdSupplier = worldIdSupplier;
        this.backend.connect(this);
    }

    public void submitChat(String message) {
        submitChat(message, null);
    }

    public void submitChat(String message, String explicitSessionId) {
        String worldId;
        try {
            worldId = worldIdSupplier.get();
        } catch (Exception error) {
            statusSink.accept("chat submit failed: " + error.getMessage());
            return;
        }

        String sessionIdForRequest = explicitSessionId;
        if (sessionIdForRequest == null || sessionIdForRequest.isBlank()) {
            sessionIdForRequest = activeSessionId;
        }
        String finalSessionIdForRequest = sessionIdForRequest;

        statusSink.accept("thinking...");
        workerExecutor.execute(() -> {
            try {
                backend.submitChatMessage(message, clientId, worldId, finalSessionIdForRequest);
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("chat submit failed: " + error.getMessage()));
            }
        });
    }

    public void createSession() {
        String worldId;
        try {
            worldId = worldIdSupplier.get();
        } catch (Exception error) {
            statusSink.accept("session new failed: " + error.getMessage());
            return;
        }

        statusSink.accept("creating session...");
        workerExecutor.execute(() -> {
            try {
                String sessionId = backend.createSession(clientId, worldId);
                activeSessionId = sessionId;
                String message = "active session: " + sessionId;
                mainExecutor.execute(() -> statusSink.accept(message));
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("session new failed: " + error.getMessage()));
            }
        });
    }

    public void listSessions() {
        String worldId;
        try {
            worldId = worldIdSupplier.get();
        } catch (Exception error) {
            statusSink.accept("session list failed: " + error.getMessage());
            return;
        }

        statusSink.accept("loading sessions...");
        workerExecutor.execute(() -> {
            try {
                List<String> sessions = backend.listSessions(clientId, worldId);
                String message;
                if (sessions.isEmpty()) {
                    message = "No sessions";
                } else {
                    List<String> rendered = new ArrayList<>(sessions.size());
                    for (String sessionId : sessions) {
                        if (sessionId.equals(activeSessionId)) {
                            rendered.add("*" + sessionId);
                        } else {
                            rendered.add(sessionId);
                        }
                    }
                    message = String.join(", ", rendered);
                }
                String statusMessage = message;
                mainExecutor.execute(() -> statusSink.accept(statusMessage));
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("session list failed: " + error.getMessage()));
            }
        });
    }

    public void switchSession(String sessionId) {
        String worldId;
        try {
            worldId = worldIdSupplier.get();
        } catch (Exception error) {
            statusSink.accept("session switch failed: " + error.getMessage());
            return;
        }

        statusSink.accept("switching session...");
        workerExecutor.execute(() -> {
            try {
                backend.switchSession(clientId, worldId, sessionId);
                activeSessionId = sessionId;
                mainExecutor.execute(() -> statusSink.accept("active session: " + sessionId));
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("session switch failed: " + error.getMessage()));
            }
        });
    }

    @Override
    public void onStatus(String jobId, String stage, String message) {
        mainExecutor.execute(() -> statusSink.accept(stage + ": " + message));
    }

    @Override
    public void onError(String jobId, String code, String message) {
        mainExecutor.execute(() -> statusSink.accept(code + ": " + message));
    }
}
