package dev.browsecraft.mod;

public interface BuildBackendListener {
    void onStatus(String jobId, String stage, String message);

    void onError(String jobId, String code, String message);
}
