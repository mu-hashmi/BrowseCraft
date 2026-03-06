package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;
import org.lwjgl.glfw.GLFW;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HudInputCaptureTest {
    @Test
    void inputModeConsumesKeyPressesForHudEditing() {
        assertTrue(HudInputCapture.shouldConsumeKey(HudChatState.Mode.INPUT, false, GLFW.GLFW_PRESS));
        assertTrue(HudInputCapture.shouldConsumeKey(HudChatState.Mode.INPUT, false, GLFW.GLFW_REPEAT));
    }

    @Test
    void inputModeDoesNotConsumeKeysWhenAnotherScreenIsOpen() {
        assertFalse(HudInputCapture.shouldConsumeKey(HudChatState.Mode.INPUT, true, GLFW.GLFW_PRESS));
    }

    @Test
    void nonInputModesDoNotConsumeKeyPresses() {
        assertFalse(HudInputCapture.shouldConsumeKey(HudChatState.Mode.HUD, false, GLFW.GLFW_PRESS));
        assertFalse(HudInputCapture.shouldConsumeKey(HudChatState.Mode.HIDDEN, false, GLFW.GLFW_PRESS));
    }

    @Test
    void printableCharsAreConsumedOnlyInHudInputMode() {
        assertTrue(HudInputCapture.shouldConsumeChar(HudChatState.Mode.INPUT, false, 'b'));
        assertFalse(HudInputCapture.shouldConsumeChar(HudChatState.Mode.INPUT, true, 'b'));
        assertFalse(HudInputCapture.shouldConsumeChar(HudChatState.Mode.HUD, false, 'b'));
        assertFalse(HudInputCapture.shouldConsumeChar(HudChatState.Mode.INPUT, false, '\n'));
    }
}
