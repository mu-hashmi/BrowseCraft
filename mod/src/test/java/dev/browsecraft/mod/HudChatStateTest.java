package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class HudChatStateTest {
    @Test
    void restoreAppliesHiddenHudSnapshot() {
        HudChatState state = new HudChatState();
        HudChatState.Snapshot snapshot = new HudChatState.Snapshot(HudChatState.Mode.HUD, "", 0);

        state.restore(snapshot);

        assertEquals(HudChatState.Mode.HUD, state.mode());
        assertEquals("", state.inputText());
        assertEquals(0, state.cursor());
    }

    @Test
    void restorePreservesInputSnapshotUsedForHudCaptures() {
        HudChatState state = new HudChatState();
        HudChatState.Snapshot snapshot = new HudChatState.Snapshot(
                HudChatState.Mode.INPUT,
                "/chat add lanterns to the arch",
                10
        );

        state.restore(snapshot);

        assertEquals(HudChatState.Mode.INPUT, state.mode());
        assertEquals("/chat add lanterns to the arch", state.inputText());
        assertEquals(10, state.cursor());
        assertEquals(snapshot, state.snapshot());
    }
}
