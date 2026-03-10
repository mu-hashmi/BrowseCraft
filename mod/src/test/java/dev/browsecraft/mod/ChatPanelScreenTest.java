package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;
import org.lwjgl.glfw.GLFW;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatPanelScreenTest {
    @Test
    void enterKeysSubmitPanelInput() {
        assertTrue(ChatPanelScreen.isSubmitKey(GLFW.GLFW_KEY_ENTER));
        assertTrue(ChatPanelScreen.isSubmitKey(GLFW.GLFW_KEY_KP_ENTER));
        assertFalse(ChatPanelScreen.isSubmitKey(GLFW.GLFW_KEY_ESCAPE));
    }

    @Test
    void scrollDeltaMatchesWheelDirection() {
        assertEquals(1, ChatPanelScreen.scrollDelta(1.0));
        assertEquals(-1, ChatPanelScreen.scrollDelta(-1.0));
        assertEquals(0, ChatPanelScreen.scrollDelta(0.0));
    }
}
