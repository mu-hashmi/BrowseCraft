package dev.browsecraft.mod;

public record BackendEndpoints(String baseUrl, String wsUrl) {
    public static BackendEndpoints localhost(String clientId) {
        return new BackendEndpoints("http://127.0.0.1:8080", "ws://127.0.0.1:8080/v1/ws/" + clientId);
    }
}
