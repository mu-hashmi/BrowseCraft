package dev.browsecraft.mod;

import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.gui.screen.Screen;
import net.minecraft.client.gui.widget.TextFieldWidget;
import net.minecraft.client.input.KeyInput;
import net.minecraft.text.OrderedText;
import net.minecraft.text.Text;
import org.lwjgl.glfw.GLFW;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Supplier;

public final class ChatPanelScreen extends Screen {
    private static final int PANEL_MARGIN = 12;
    private static final int PANEL_WIDTH = 360;
    private static final int PANEL_HEADER_HEIGHT = 22;
    private static final int PANEL_FOOTER_HEIGHT = 36;
    private static final int PANEL_INSET = 8;

    private final Consumer<String> submitHandler;
    private final Supplier<List<ChatMessage>> messagesSupplier;
    private final Supplier<String> activitySupplier;
    private final String initialInput;
    private TextFieldWidget inputField;
    private int scrollLines;

    public ChatPanelScreen(
            Consumer<String> submitHandler,
            Supplier<List<ChatMessage>> messagesSupplier,
            Supplier<String> activitySupplier,
            String initialInput
    ) {
        super(Text.literal("BrowseCraft Chat"));
        this.submitHandler = submitHandler;
        this.messagesSupplier = messagesSupplier;
        this.activitySupplier = activitySupplier;
        this.initialInput = initialInput;
    }

    @Override
    protected void init() {
        int panelLeft = panelLeft();
        int panelRight = panelRight();
        int panelBottom = panelBottom();
        int inputY = panelBottom - PANEL_FOOTER_HEIGHT + 8;
        this.inputField = new TextFieldWidget(
                this.textRenderer,
                panelLeft + PANEL_INSET,
                inputY,
                (panelRight - panelLeft) - (PANEL_INSET * 2),
                20,
                Text.literal("Message")
        );
        this.inputField.setMaxLength(1024);
        this.inputField.setText(this.initialInput);
        this.inputField.setDrawsBackground(true);
        this.addDrawableChild(this.inputField);
        this.setInitialFocus(this.inputField);
    }

    @Override
    public boolean keyPressed(KeyInput input) {
        int keyCode = input.getKeycode();
        if (isSubmitKey(keyCode)) {
            submitInput();
            return true;
        }
        return super.keyPressed(input);
    }

    @Override
    public boolean mouseScrolled(double mouseX, double mouseY, double horizontalAmount, double verticalAmount) {
        int delta = scrollDelta(verticalAmount);
        if (delta != 0) {
            this.scrollLines += delta;
            clampScroll();
            return true;
        }
        return super.mouseScrolled(mouseX, mouseY, horizontalAmount, verticalAmount);
    }

    @Override
    public boolean shouldPause() {
        return false;
    }

    @Override
    public void render(DrawContext context, int mouseX, int mouseY, float delta) {
        context.fill(0, 0, this.width, this.height, 0x40000000);

        int left = panelLeft();
        int right = panelRight();
        int top = panelTop();
        int bottom = panelBottom();
        context.fill(left, top, right, bottom, 0xB0101010);

        String activity = this.activitySupplier.get();
        if (activity != null && !activity.isBlank()) {
            context.drawText(this.textRenderer, activity, left + PANEL_INSET, top + 7, 0xFFE8E8A0, false);
        } else {
            context.drawText(this.textRenderer, "BrowseCraft Chat", left + PANEL_INSET, top + 7, 0xFFFFFFFF, false);
        }

        int messagesTop = top + PANEL_HEADER_HEIGHT;
        int messagesBottom = bottom - PANEL_FOOTER_HEIGHT;
        int textX = left + PANEL_INSET;
        int textWidth = (right - left) - (PANEL_INSET * 2);

        List<RenderedLine> renderedLines = buildRenderedLines(this.messagesSupplier.get(), this.textRenderer, textWidth);
        int lineHeight = this.textRenderer.fontHeight + 2;
        int maxVisible = Math.max(1, (messagesBottom - messagesTop) / lineHeight);
        int maxScroll = Math.max(0, renderedLines.size() - maxVisible);
        this.scrollLines = Math.clamp(this.scrollLines, 0, maxScroll);
        int start = Math.max(0, renderedLines.size() - maxVisible - this.scrollLines);
        int end = Math.min(renderedLines.size(), start + maxVisible);
        int y = messagesTop + 2;
        for (int index = start; index < end; index++) {
            RenderedLine line = renderedLines.get(index);
            context.drawText(this.textRenderer, line.text(), textX, y, line.color(), false);
            y += lineHeight;
        }

        if (this.inputField != null) {
            this.inputField.render(context, mouseX, mouseY, delta);
        }
    }

    private void submitInput() {
        if (this.inputField == null) {
            return;
        }
        String message = this.inputField.getText().trim();
        if (message.isEmpty()) {
            return;
        }
        this.submitHandler.accept(message);
        this.inputField.setText("");
        this.scrollLines = 0;
    }

    static boolean isSubmitKey(int keyCode) {
        return keyCode == GLFW.GLFW_KEY_ENTER || keyCode == GLFW.GLFW_KEY_KP_ENTER;
    }

    static int scrollDelta(double verticalAmount) {
        if (verticalAmount > 0) {
            return 1;
        }
        if (verticalAmount < 0) {
            return -1;
        }
        return 0;
    }

    private void clampScroll() {
        List<RenderedLine> renderedLines = buildRenderedLines(
                this.messagesSupplier.get(),
                this.textRenderer,
                panelWidth() - (PANEL_INSET * 2)
        );
        int lineHeight = this.textRenderer.fontHeight + 2;
        int maxVisible = Math.max(1, (panelHeight() - PANEL_HEADER_HEIGHT - PANEL_FOOTER_HEIGHT) / lineHeight);
        int maxScroll = Math.max(0, renderedLines.size() - maxVisible);
        this.scrollLines = Math.clamp(this.scrollLines, 0, maxScroll);
    }

    private static List<RenderedLine> buildRenderedLines(List<ChatMessage> messages, TextRenderer textRenderer, int width) {
        List<RenderedLine> lines = new ArrayList<>();
        for (ChatMessage message : messages) {
            int color = message.role() == ChatRole.USER ? 0xFF7FD7FF : 0xFFF0F0F0;
            String prefix = message.role() == ChatRole.USER ? "You: " : "AI: ";
            Text text = Text.literal(prefix + message.text());
            List<OrderedText> wrapped = textRenderer.wrapLines(text, width);
            if (wrapped.isEmpty()) {
                lines.add(new RenderedLine(Text.literal(prefix).asOrderedText(), color));
                continue;
            }
            for (OrderedText wrappedLine : wrapped) {
                lines.add(new RenderedLine(wrappedLine, color));
            }
        }
        return lines;
    }

    private int panelLeft() {
        return this.width - panelWidth() - PANEL_MARGIN;
    }

    private int panelRight() {
        return this.width - PANEL_MARGIN;
    }

    private int panelTop() {
        return PANEL_MARGIN;
    }

    private int panelBottom() {
        return this.height - PANEL_MARGIN;
    }

    private int panelWidth() {
        return Math.min(PANEL_WIDTH, this.width - (PANEL_MARGIN * 2));
    }

    private int panelHeight() {
        return this.height - (PANEL_MARGIN * 2);
    }

    public enum ChatRole {
        USER,
        ASSISTANT
    }

    public record ChatMessage(ChatRole role, String text) {
    }

    private record RenderedLine(OrderedText text, int color) {
    }
}
