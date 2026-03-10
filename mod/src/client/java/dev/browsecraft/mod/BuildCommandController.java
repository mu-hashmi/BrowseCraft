package dev.browsecraft.mod;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.function.Consumer;
import java.util.function.Supplier;

public final class BuildCommandController implements BuildBackendListener {
    public interface ChatEventListener {
        void onUserMessage(String message);

        void onAssistantDelta(String delta);

        void onAssistantMessage(String message);

        void onToolStatus(String status);
    }

    private final String clientId;
    private final BuildBackend backend;
    private final OverlayState overlayState;
    private final Executor mainExecutor;
    private final Executor workerExecutor;
    private final Consumer<String> statusSink;
    private final Supplier<String> worldIdSupplier;
    private final ChatEventListener chatEvents;
    private String activeSessionId;

    public BuildCommandController(
            String clientId,
            BuildBackend backend,
            OverlayState overlayState,
            Executor mainExecutor,
            Executor workerExecutor,
            Consumer<String> statusSink,
            Supplier<String> worldIdSupplier,
            ChatEventListener chatEvents
    ) {
        this.clientId = clientId;
        this.backend = backend;
        this.overlayState = overlayState;
        this.mainExecutor = mainExecutor;
        this.workerExecutor = workerExecutor;
        this.statusSink = statusSink;
        this.worldIdSupplier = worldIdSupplier;
        this.chatEvents = chatEvents;
        this.backend.connect(this);
    }

    public void submitChat(String message) {
        submitChat(message, null, "build");
    }

    public void submitChat(String message, String explicitSessionId) {
        submitChat(message, explicitSessionId, "build");
    }

    public void submitPlan(String message) {
        submitChat(message, null, "plan");
    }

    public void submitPlan(String message, String explicitSessionId) {
        submitChat(message, explicitSessionId, "plan");
    }

    private void submitChat(String message, String explicitSessionId, String mode) {
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
        mainExecutor.execute(() -> chatEvents.onUserMessage(message));
        workerExecutor.execute(() -> {
            try {
                backend.submitChatMessage(message, clientId, worldId, finalSessionIdForRequest, mode);
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
        mainExecutor.execute(() -> {
            switch (stage) {
                case "chat.delta" -> chatEvents.onAssistantDelta(message);
                case "chat.response" -> {
                    chatEvents.onAssistantMessage(message);
                    statusSink.accept("chat: " + message);
                }
                case "tool_status" -> {
                    chatEvents.onToolStatus(message);
                    statusSink.accept(message);
                }
                default -> statusSink.accept(stage + ": " + message);
            }
        });
    }

    @Override
    public void onError(String jobId, String code, String message) {
        mainExecutor.execute(() -> statusSink.accept(code + ": " + message));
    }

    public void submitSearch(String query) {
        statusSink.accept("searching...");
        mainExecutor.execute(() -> chatEvents.onUserMessage("/search " + query));
        workerExecutor.execute(() -> {
            try {
                backend.submitSearch(clientId, query);
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("search failed: " + error.getMessage()));
            }
        });
    }

    public void submitImagine(String prompt) {
        statusSink.accept("imagining...");
        mainExecutor.execute(() -> chatEvents.onUserMessage("/imagine " + prompt));
        workerExecutor.execute(() -> {
            try {
                backend.submitImagine(clientId, prompt);
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("imagine failed: " + error.getMessage()));
            }
        });
    }
}
