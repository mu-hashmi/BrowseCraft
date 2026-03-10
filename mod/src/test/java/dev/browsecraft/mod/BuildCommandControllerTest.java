package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BuildCommandControllerTest {
    private static final BuildCommandController.ChatEventListener NO_OP_CHAT_EVENTS = new BuildCommandController.ChatEventListener() {
        @Override
        public void onUserMessage(String message) {
        }

        @Override
        public void onAssistantDelta(String delta) {
        }

        @Override
        public void onAssistantMessage(String message) {
        }

        @Override
        public void onToolStatus(String status) {
        }
    };

    @Test
    void chatUsesActiveSessionCreatedBySessionNew() {
        FakeBackend backend = new FakeBackend();
        List<String> statuses = new ArrayList<>();
        BuildCommandController controller = new BuildCommandController(
                "client-1",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                statuses::add,
                () -> "world-1",
                NO_OP_CHAT_EVENTS
        );

        controller.createSession();
        controller.submitChat("hello");

        assertEquals("world-1", backend.lastChatWorldId);
        assertEquals("session-1", backend.lastChatSessionId);
        assertEquals("hello", backend.lastChatMessage);
        assertEquals("build", backend.lastChatMode);
        assertTrue(statuses.contains("active session: session-1"));
    }

    @Test
    void explicitChatSessionOverridesActiveSession() {
        FakeBackend backend = new FakeBackend();
        BuildCommandController controller = new BuildCommandController(
                "client-2",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                message -> {
                },
                () -> "world-2",
                NO_OP_CHAT_EVENTS
        );

        controller.createSession();
        controller.submitChat("hello", "session-explicit");

        assertEquals("world-2", backend.lastChatWorldId);
        assertEquals("session-explicit", backend.lastChatSessionId);
    }

    @Test
    void sessionSwitchChangesActiveSession() {
        FakeBackend backend = new FakeBackend();
        backend.sessions = List.of("session-a", "session-b");
        BuildCommandController controller = new BuildCommandController(
                "client-3",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                message -> {
                },
                () -> "world-3",
                NO_OP_CHAT_EVENTS
        );

        controller.switchSession("session-b");
        controller.submitChat("hello");

        assertEquals("client-3", backend.lastSwitchClientId);
        assertEquals("world-3", backend.lastSwitchWorldId);
        assertEquals("session-b", backend.lastSwitchedSessionId);
        assertEquals("session-b", backend.lastChatSessionId);
    }

    @Test
    void sessionListMarksActiveSession() {
        FakeBackend backend = new FakeBackend();
        backend.sessions = List.of("session-a", "session-b");
        List<String> statuses = new ArrayList<>();
        BuildCommandController controller = new BuildCommandController(
                "client-4",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                statuses::add,
                () -> "world-4",
                NO_OP_CHAT_EVENTS
        );

        controller.switchSession("session-a");
        controller.listSessions();

        assertTrue(statuses.contains("*session-a, session-b"));
    }

    @Test
    void planSubmissionUsesPlanMode() {
        FakeBackend backend = new FakeBackend();
        BuildCommandController controller = new BuildCommandController(
                "client-5",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                message -> {
                },
                () -> "world-5",
                NO_OP_CHAT_EVENTS
        );

        controller.submitPlan("plan a tower");

        assertEquals("plan", backend.lastChatMode);
        assertEquals("plan a tower", backend.lastChatMessage);
    }

    @Test
    void searchCommandRoutesToBackend() {
        FakeBackend backend = new FakeBackend();
        BuildCommandController controller = new BuildCommandController(
                "client-search",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                message -> {
                },
                () -> "world-search",
                NO_OP_CHAT_EVENTS
        );

        controller.submitSearch("medieval castle");

        assertEquals("client-search", backend.lastSearchClientId);
        assertEquals("medieval castle", backend.lastSearchQuery);
    }

    @Test
    void imagineCommandRoutesToBackend() {
        FakeBackend backend = new FakeBackend();
        BuildCommandController controller = new BuildCommandController(
                "client-imagine",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                message -> {
                },
                () -> "world-imagine",
                NO_OP_CHAT_EVENTS
        );

        controller.submitImagine("dragon statue");

        assertEquals("client-imagine", backend.lastImagineClientId);
        assertEquals("dragon statue", backend.lastImaginePrompt);
    }

    @Test
    void backendStatusStagesReachChatEventListener() {
        FakeBackend backend = new FakeBackend();
        RecordingChatEvents chatEvents = new RecordingChatEvents();
        List<String> statuses = new ArrayList<>();
        BuildCommandController controller = new BuildCommandController(
                "client-events",
                backend,
                new OverlayState(),
                directExecutor(),
                directExecutor(),
                statuses::add,
                () -> "world-events",
                chatEvents
        );

        controller.onStatus("", "chat.delta", "stream ");
        controller.onStatus("", "chat.response", "final");
        controller.onStatus("", "tool_status", "🔍 Inspecting area...");

        assertEquals(List.of("stream "), chatEvents.assistantDeltas);
        assertEquals(List.of("final"), chatEvents.assistantMessages);
        assertEquals(List.of("🔍 Inspecting area..."), chatEvents.toolStatuses);
        assertTrue(statuses.contains("chat: final"));
        assertTrue(statuses.contains("🔍 Inspecting area..."));
    }

    private Executor directExecutor() {
        return Runnable::run;
    }

    private static final class FakeBackend implements BuildBackend {
        private List<String> sessions = List.of("session-1");
        private String lastChatMessage;
        private String lastChatWorldId;
        private String lastChatSessionId;
        private String lastChatMode;
        private String lastCreateClientId;
        private String lastCreateWorldId;
        private String lastListClientId;
        private String lastListedWorldId;
        private String lastSwitchClientId;
        private String lastSwitchWorldId;
        private String lastSwitchedSessionId;
        private String lastSearchClientId;
        private String lastSearchQuery;
        private String lastImagineClientId;
        private String lastImaginePrompt;

        @Override
        public void connect(BuildBackendListener listener) {
        }

        @Override
        public void submitChatMessage(String message, String clientId, String worldId, String sessionId, String mode) {
            this.lastChatMessage = message;
            this.lastChatWorldId = worldId;
            this.lastChatSessionId = sessionId;
            this.lastChatMode = mode;
        }

        @Override
        public String createSession(String clientId, String worldId) {
            this.lastCreateClientId = clientId;
            this.lastCreateWorldId = worldId;
            return "session-1";
        }

        @Override
        public List<String> listSessions(String clientId, String worldId) {
            this.lastListClientId = clientId;
            this.lastListedWorldId = worldId;
            return sessions;
        }

        @Override
        public void switchSession(String clientId, String worldId, String sessionId) {
            this.lastSwitchClientId = clientId;
            this.lastSwitchWorldId = worldId;
            this.lastSwitchedSessionId = sessionId;
        }

        @Override
        public void submitSearch(String clientId, String query) {
            this.lastSearchClientId = clientId;
            this.lastSearchQuery = query;
        }

        @Override
        public void submitImagine(String clientId, String prompt) {
            this.lastImagineClientId = clientId;
            this.lastImaginePrompt = prompt;
        }

        @Override
        public void close() {
        }
    }

    private static final class RecordingChatEvents implements BuildCommandController.ChatEventListener {
        private final List<String> userMessages = new ArrayList<>();
        private final List<String> assistantDeltas = new ArrayList<>();
        private final List<String> assistantMessages = new ArrayList<>();
        private final List<String> toolStatuses = new ArrayList<>();

        @Override
        public void onUserMessage(String message) {
            userMessages.add(message);
        }

        @Override
        public void onAssistantDelta(String delta) {
            assistantDeltas.add(delta);
        }

        @Override
        public void onAssistantMessage(String message) {
            assistantMessages.add(message);
        }

        @Override
        public void onToolStatus(String status) {
            toolStatuses.add(status);
        }
    }
}
