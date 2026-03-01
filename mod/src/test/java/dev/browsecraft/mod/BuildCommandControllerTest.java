package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BuildCommandControllerTest {

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
                () -> "world-1"
        );

        controller.createSession();
        controller.submitChat("hello");

        assertEquals("world-1", backend.lastChatWorldId);
        assertEquals("session-1", backend.lastChatSessionId);
        assertEquals("hello", backend.lastChatMessage);
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
                () -> "world-2"
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
                () -> "world-3"
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
                () -> "world-4"
        );

        controller.switchSession("session-a");
        controller.listSessions();

        assertTrue(statuses.contains("*session-a, session-b"));
    }

    private Executor directExecutor() {
        return Runnable::run;
    }

    private static final class FakeBackend implements BuildBackend {
        private List<String> sessions = List.of("session-1");
        private String lastChatMessage;
        private String lastChatWorldId;
        private String lastChatSessionId;
        private String lastCreateClientId;
        private String lastCreateWorldId;
        private String lastListClientId;
        private String lastListedWorldId;
        private String lastSwitchClientId;
        private String lastSwitchWorldId;
        private String lastSwitchedSessionId;

        @Override
        public void connect(BuildBackendListener listener) {
        }

        @Override
        public void submitChatMessage(String message, String clientId, String worldId, String sessionId) {
            this.lastChatMessage = message;
            this.lastChatWorldId = worldId;
            this.lastChatSessionId = sessionId;
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
        public void close() {
        }
    }
}
