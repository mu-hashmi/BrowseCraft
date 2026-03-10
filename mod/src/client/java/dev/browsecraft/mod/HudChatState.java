package dev.browsecraft.mod;

final class HudChatState {
    enum Mode {
        HIDDEN,
        HUD,
        INPUT
    }

    record Snapshot(Mode mode, String inputText, int cursor) {
    }

    private final StringBuilder input = new StringBuilder();
    private Mode mode = Mode.HIDDEN;
    private int cursor;

    Mode mode() {
        return mode;
    }

    String inputText() {
        return input.toString();
    }

    int cursor() {
        return cursor;
    }

    Snapshot snapshot() {
        return new Snapshot(mode, input.toString(), cursor);
    }

    void restore(Snapshot snapshot) {
        mode = snapshot.mode();
        input.setLength(0);
        input.append(snapshot.inputText());
        cursor = snapshot.cursor();
    }
}
