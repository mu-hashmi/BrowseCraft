package dev.browsecraft.mod;

import java.io.IOException;
import java.util.List;

public interface BuildBackend {
    void connect(BuildBackendListener listener);

    String submitBuildQuery(String query, String clientId) throws IOException, InterruptedException;

    String submitImaginePrompt(String prompt, String clientId) throws IOException, InterruptedException;

    String submitImagineModifyPrompt(String prompt, String clientId) throws IOException, InterruptedException;

    void submitChatMessage(
            String message,
            String clientId,
            String worldId,
            String sessionId
    ) throws IOException, InterruptedException;

    String createSession(String clientId, String worldId) throws IOException, InterruptedException;

    List<String> listSessions(String clientId, String worldId) throws IOException, InterruptedException;

    void switchSession(String clientId, String worldId, String sessionId) throws IOException, InterruptedException;

    void close();
}
