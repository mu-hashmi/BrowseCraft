package dev.browsecraft.mod;

final class HudInputCapture {
    private HudInputCapture() {
    }

    static boolean shouldConsumeKey(HudChatState.Mode mode, boolean hasScreen, int action) {
        return mode == HudChatState.Mode.INPUT
                && !hasScreen
                && (action == org.lwjgl.glfw.GLFW.GLFW_PRESS || action == org.lwjgl.glfw.GLFW.GLFW_REPEAT);
    }

    static boolean shouldConsumeChar(HudChatState.Mode mode, boolean hasScreen, int codepoint) {
        return mode == HudChatState.Mode.INPUT
                && !hasScreen
                && Character.isValidCodePoint(codepoint)
                && !Character.isISOControl(codepoint);
    }
}
