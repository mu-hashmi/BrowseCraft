package dev.browsecraft.mod;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicReference;
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
    private final AtomicReference<String> activeJobId = new AtomicReference<>();
    private final AtomicReference<String> activeSessionId = new AtomicReference<>();

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

    public void submit(String query) {
        statusSink.accept("searching...");
        workerExecutor.execute(() -> {
            try {
                String jobId = backend.submitBuildQuery(query, clientId);
                activeJobId.set(jobId);
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("build submit failed: " + error.getMessage()));
            }
        });
    }

    public void submitImagine(String prompt) {
        statusSink.accept("generating image...");
        workerExecutor.execute(() -> {
            try {
                String jobId = backend.submitImaginePrompt(prompt, clientId);
                activeJobId.set(jobId);
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("imagine submit failed: " + error.getMessage()));
            }
        });
    }

    public void submitImagineModify(String prompt) {
        statusSink.accept("editing image...");
        workerExecutor.execute(() -> {
            try {
                String jobId = backend.submitImagineModifyPrompt(prompt, clientId);
                activeJobId.set(jobId);
            } catch (Exception error) {
                mainExecutor.execute(() -> statusSink.accept("imagine modify failed: " + error.getMessage()));
            }
        });
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

        String sessionId = explicitSessionId;
        if (sessionId == null || sessionId.isBlank()) {
            sessionId = activeSessionId.get();
        }
        String sessionIdForRequest = sessionId;

        statusSink.accept("thinking...");
        workerExecutor.execute(() -> {
            try {
                backend.submitChatMessage(message, clientId, worldId, sessionIdForRequest);
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
                activeSessionId.set(sessionId);
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
                String active = activeSessionId.get();
                String message;
                if (sessions.isEmpty()) {
                    message = "No sessions";
                } else {
                    List<String> rendered = new ArrayList<>(sessions.size());
                    for (String sessionId : sessions) {
                        if (sessionId.equals(active)) {
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
                activeSessionId.set(sessionId);
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
    public void onReady(String jobId, String sourceType, String sourceUrl, double confidence, BuildPlan plan) {
        activeJobId.compareAndSet(jobId, null);
        mainExecutor.execute(() -> {
            overlayState.setPlan(plan);
            statusSink.accept("ready from " + sourceType + " (" + plan.totalBlocks() + " blocks)");
        });
    }

    @Override
    public void onError(String jobId, String code, String message) {
        if (jobId != null && !jobId.isEmpty()) {
            activeJobId.compareAndSet(jobId, null);
        }
        mainExecutor.execute(() -> statusSink.accept(code + ": " + message));
    }
}
