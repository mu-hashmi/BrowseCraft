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

    Mode cycleMode() {
        mode = switch (mode) {
            case HIDDEN -> Mode.HUD;
            case HUD -> Mode.INPUT;
            case INPUT -> Mode.HIDDEN;
        };
        if (mode != Mode.INPUT) {
            clearInput();
        }
        return mode;
    }

    void openInput(String prefill) {
        mode = Mode.INPUT;
        input.setLength(0);
        input.append(prefill);
        cursor = input.length();
    }

    void ensureHudVisible() {
        if (mode == Mode.HIDDEN) {
            mode = Mode.HUD;
        }
    }

    void insert(char value) {
        input.insert(cursor, value);
        cursor++;
    }

    void backspace() {
        if (cursor == 0) {
            return;
        }
        input.deleteCharAt(cursor - 1);
        cursor--;
    }

    void moveLeft() {
        if (cursor > 0) {
            cursor--;
        }
    }

    void moveRight() {
        if (cursor < input.length()) {
            cursor++;
        }
    }

    String submit() {
        String message = input.toString().trim();
        clearInput();
        mode = Mode.HUD;
        return message;
    }

    void cancelInput() {
        clearInput();
        mode = Mode.HUD;
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

    private void clearInput() {
        input.setLength(0);
        cursor = 0;
    }
}
