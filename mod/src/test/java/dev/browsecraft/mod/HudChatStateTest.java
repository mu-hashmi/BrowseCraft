package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HudChatStateTest {
    @Test
    void bCycleFollowsHiddenHudInputOrder() {
        HudChatState state = new HudChatState();

        assertEquals(HudChatState.Mode.HIDDEN, state.mode());
        assertEquals(HudChatState.Mode.HUD, state.cycleMode());
        assertEquals(HudChatState.Mode.INPUT, state.cycleMode());
        assertEquals(HudChatState.Mode.HIDDEN, state.cycleMode());
    }

    @Test
    void inputEditorSupportsInsertMoveAndBackspace() {
        HudChatState state = new HudChatState();
        state.openInput("");
        state.insert('h');
        state.insert('l');
        state.moveLeft();
        state.insert('e');
        state.moveRight();
        state.insert('o');
        state.backspace();
        state.insert('o');

        assertEquals("helo", state.inputText());
    }

    @Test
    void submitReturnsTrimmedMessageAndSwitchesToHud() {
        HudChatState state = new HudChatState();
        state.openInput("  build house  ");

        String message = state.submit();

        assertEquals("build house", message);
        assertEquals(HudChatState.Mode.HUD, state.mode());
        assertEquals("", state.inputText());
    }

    @Test
    void cancelClearsInputAndReturnsToHud() {
        HudChatState state = new HudChatState();
        state.openInput("abc");

        state.cancelInput();

        assertEquals(HudChatState.Mode.HUD, state.mode());
        assertEquals("", state.inputText());
        assertEquals(0, state.cursor());
    }

    @Test
    void snapshotRestorePreservesModeInputAndCursor() {
        HudChatState state = new HudChatState();
        state.openInput("stone wall");
        state.moveLeft();
        state.moveLeft();

        HudChatState.Snapshot snapshot = state.snapshot();

        state.cancelInput();
        state.restore(snapshot);

        assertEquals(HudChatState.Mode.INPUT, state.mode());
        assertEquals("stone wall", state.inputText());
        assertEquals("stone wall".length() - 2, state.cursor());
    }
}
