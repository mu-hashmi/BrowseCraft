package dev.browsecraft.mod;

import java.io.IOException;
import java.util.List;

public interface BuildBackend {
    void connect(BuildBackendListener listener);

    void submitChatMessage(
            String message,
            String clientId,
            String worldId,
            String sessionId,
            String mode
    ) throws IOException, InterruptedException;

    String createSession(String clientId, String worldId) throws IOException, InterruptedException;

    List<String> listSessions(String clientId, String worldId) throws IOException, InterruptedException;

    void switchSession(String clientId, String worldId, String sessionId) throws IOException, InterruptedException;

    void submitSearch(String clientId, String query) throws IOException, InterruptedException;

    void submitImagine(String clientId, String prompt) throws IOException, InterruptedException;

    void close();
}
