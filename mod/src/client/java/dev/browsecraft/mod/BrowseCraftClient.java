package dev.browsecraft.mod;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.hud.VanillaHudElements;
import net.minecraft.block.BlockState;
import net.minecraft.client.font.TextRenderer;
import net.minecraft.client.gui.DrawContext;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.client.util.ScreenshotRecorder;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.item.BlockItem;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.text.OrderedText;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import net.minecraft.util.WorldSavePath;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import org.lwjgl.glfw.GLFW;
import org.lwjgl.glfw.GLFWCharCallbackI;
import org.lwjgl.glfw.GLFWKeyCallbackI;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.argument;
import static net.fabricmc.fabric.api.client.command.v2.ClientCommandManager.literal;

public final class BrowseCraftClient implements ClientModInitializer {
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final int CHAT_BUILD_TEST_TIMEOUT_TICKS = 20 * 90;
    private static final int CHAT_BUILD_TEST_SETTLE_TICKS = 20;
    private static final int HUD_TEST_SETTLE_TICKS = 4;
    private static final int HUD_MARGIN = 10;
    private static final int HUD_PANEL_WIDTH = 340;
    private static final int HUD_PANEL_HEIGHT = 210;
    private static final int HUD_HEADER_HEIGHT = 20;
    private static final int HUD_INPUT_HEIGHT = 24;
    private static final int HUD_INSET = 7;
    private static final Identifier HUD_LAYER_ID = Identifier.of("browsecraft", "hud_panel");
    private static final String HUD_TEST_ON_STARTUP_PROPERTY = "browsecraft.hudTestOnStartup";
    private static final String HUD_TEST_EXIT_ON_COMPLETE_PROPERTY = "browsecraft.hudTestExitOnComplete";
    private static final Set<String> TERRAIN_BLOCK_IDS = Set.of(
            "minecraft:grass_block",
            "minecraft:dirt",
            "minecraft:coarse_dirt",
            "minecraft:podzol",
            "minecraft:mycelium",
            "minecraft:rooted_dirt",
            "minecraft:bedrock",
            "minecraft:sand",
            "minecraft:red_sand",
            "minecraft:gravel",
            "minecraft:deepslate",
            "minecraft:tuff"
    );
    private static volatile BrowseCraftClient instance;

    public static volatile Path latestBuildTestJsonPath;
    public static volatile Path latestBuildTestScreenshotPath;
    public static volatile Path latestHudTestJsonPath;

    private OverlayState overlayState;
    private GhostRenderer ghostRenderer;
    private BuildCommandController commandController;
    private BuildBackend backend;
    private ExecutorService workerExecutor;
    private String clientId;
    private final WorldIdResolver worldIdResolver = new WorldIdResolver();
    private boolean hasUndoState;
    private boolean undoClearsOverlay;
    private OverlayState.BlueprintState undoBlueprintState;
    private List<AbsolutePlacement> undoWorldPlacements;

    private KeyBinding openChatKey;
    private KeyBinding rotateKey;
    private KeyBinding shiftUpKey;
    private KeyBinding shiftDownKey;
    private KeyBinding confirmKey;
    private KeyBinding cancelKey;
    private KeyBinding.Category keyCategory;

    private int buildTestCaptureTicksRemaining = -1;
    private int chatBuildTestWaitTicksRemaining = -1;
    private int chatBuildTestStableTicks;
    private long overlayPlacementRevision;
    private long chatBuildTestBaselineRevision;
    private long chatBuildTestObservedRevision;
    private int buildTestStepSequence;
    private String buildTestStepId;
    private long buildTestStepStartedAtMillis = -1;
    private long buildTestStepFirstPlacementAtMillis = -1;
    private long buildTestStepReadyForCaptureAtMillis = -1;
    private int buildTestStepTickCount;
    private String buildTestTrigger = "build-test";
    private String buildTestPrompt;
    private int toolCallSequence;
    private final List<JsonObject> buildTestToolCalls = new ArrayList<>();
    private final List<JsonObject> buildTestStatusTimeline = new ArrayList<>();
    private int inventoryPollCountdown;
    private volatile String latestStatusMessage = "";
    private final HudChatState hudChatState = new HudChatState();
    private GLFWKeyCallbackI previousKeyCallback;
    private GLFWCharCallbackI previousCharCallback;
    private final List<ChatMessage> chatHistory = new ArrayList<>();
    private String activeToolStatus = "";
    private boolean assistantStreaming;
    private HudCaptureSession hudCaptureSession;
    private HudRenderSnapshot latestHudRenderSnapshot = new HudRenderSnapshot(
            HudChatState.Mode.HIDDEN,
            false,
            0,
            0,
            "",
            "",
            false,
            null,
            null,
            List.of(),
            0,
            "",
            0,
            0
    );
    private boolean hudTestStartupTriggered;

    @Override
    public void onInitializeClient() {
        instance = this;
        this.clientId = UUID.randomUUID().toString();
        this.overlayState = new OverlayState();
        this.ghostRenderer = new GhostRenderer(overlayState);
        this.backend = new BackendClient(BackendEndpoints.localhost(clientId), "1.21.11", this::handleToolRequest);
        this.workerExecutor = Executors.newVirtualThreadPerTaskExecutor();

        Executor mainExecutor = runnable -> MinecraftClient.getInstance().execute(runnable);
        Consumer<String> statusSink = message -> {
            latestStatusMessage = message;
            recordBuildTestStatusEvent(message);
            MinecraftClient client = MinecraftClient.getInstance();
            if (client.player != null) {
                client.player.sendMessage(Text.literal(message), true);
            }
        };

        this.commandController = new BuildCommandController(
                clientId,
                backend,
                overlayState,
                mainExecutor,
                workerExecutor,
                statusSink,
                () -> worldIdResolver.resolve(MinecraftClient.getInstance()),
                new BuildCommandController.ChatEventListener() {
                    @Override
                    public void onUserMessage(String message) {
                        handleChatUserMessage(message);
                    }

                    @Override
                    public void onAssistantDelta(String delta) {
                        handleChatAssistantDelta(delta);
                    }

                    @Override
                    public void onAssistantMessage(String message) {
                        handleChatAssistantMessage(message);
                    }

                    @Override
                    public void onToolStatus(String status) {
                        handleToolStatus(status);
                    }
                }
        );

        registerKeyBindings();
        registerHudInputCallbacks();
        registerCommands();
        registerClientTick();
        registerHudRenderer();
        ghostRenderer.register();
    }

    private void registerKeyBindings() {
        this.keyCategory = KeyBinding.Category.create(Identifier.of("browsecraft", "controls"));
        this.openChatKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.open_chat",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_B,
                keyCategory
        ));
        this.rotateKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.rotate",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_R,
                keyCategory
        ));
        this.shiftUpKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.shift_up",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_PAGE_UP,
                keyCategory
        ));
        this.shiftDownKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.shift_down",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_PAGE_DOWN,
                keyCategory
        ));
        this.confirmKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.confirm",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_ENTER,
                keyCategory
        ));
        this.cancelKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.cancel",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_ESCAPE,
                keyCategory
        ));
    }

    private void registerCommands() {
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            dispatcher.register(literal("build-test")
                    .executes(context -> {
                        runBuildTestCommand(MinecraftClient.getInstance());
                        return 1;
                    }));

            dispatcher.register(literal("hud-test")
                    .executes(context -> {
                        runHudTestCommand(MinecraftClient.getInstance());
                        return 1;
                    }));

            dispatcher.register(literal("chat")
                    .executes(context -> {
                        openHudInput("");
                        return 1;
                    })
                    .then(argument("message", StringArgumentType.greedyString())
                            .executes(context -> {
                                String message = StringArgumentType.getString(context, "message");
                                openHudInput(message);
                                return 1;
                            })));

            dispatcher.register(literal("plan")
                    .then(argument("message", StringArgumentType.greedyString())
                            .executes(context -> {
                                String message = StringArgumentType.getString(context, "message");
                                commandController.submitPlan(message);
                                return 1;
                            })));

            dispatcher.register(literal("search")
                    .then(argument("query", StringArgumentType.greedyString())
                            .executes(context -> {
                                String query = StringArgumentType.getString(context, "query");
                                hudChatState.ensureHudVisible();
                                commandController.submitSearch(query);
                                return 1;
                            })));

            dispatcher.register(literal("imagine")
                    .then(argument("prompt", StringArgumentType.greedyString())
                            .executes(context -> {
                                String prompt = StringArgumentType.getString(context, "prompt");
                                hudChatState.ensureHudVisible();
                                commandController.submitImagine(prompt);
                                return 1;
                            })));

            dispatcher.register(literal("session")
                    .then(literal("new")
                            .executes(context -> {
                                commandController.createSession();
                                return 1;
                            }))
                    .then(literal("list")
                            .executes(context -> {
                                commandController.listSessions();
                                return 1;
                            }))
                    .then(literal("switch")
                            .then(argument("id", StringArgumentType.word())
                                    .executes(context -> {
                                        String sessionId = StringArgumentType.getString(context, "id");
                                        commandController.switchSession(sessionId);
                                        return 1;
                                    }))));

            dispatcher.register(literal("blueprints")
                    .then(literal("save")
                            .then(argument("name", StringArgumentType.word())
                                    .executes(context -> commandSaveBlueprint(
                                            MinecraftClient.getInstance(),
                                            StringArgumentType.getString(context, "name")
                                    ))))
                    .then(literal("load")
                            .then(argument("name", StringArgumentType.word())
                                    .executes(context -> commandLoadBlueprint(
                                            MinecraftClient.getInstance(),
                                            StringArgumentType.getString(context, "name")
                                    ))))
                    .then(literal("list")
                            .executes(context -> commandListBlueprints(MinecraftClient.getInstance()))));

            dispatcher.register(literal("materials")
                    .executes(context -> commandMaterials(MinecraftClient.getInstance())));
        });
    }

    private void registerClientTick() {
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            while (openChatKey.wasPressed()) {
                hudChatState.cycleMode();
            }

            if (client.player == null || client.world == null) {
                ghostRenderer.setAvailableBlockTypes(Set.of());
                return;
            }

            if (!hudTestStartupTriggered && Boolean.getBoolean(HUD_TEST_ON_STARTUP_PROPERTY)) {
                hudTestStartupTriggered = true;
                runHudTestCommand(client);
            }

            if (hudCaptureSession != null) {
                tickHudCaptureSession(client);
            }

            if (inventoryPollCountdown <= 0) {
                refreshAvailableBlockTypes(client);
                inventoryPollCountdown = 20;
            } else {
                inventoryPollCountdown--;
            }

            if (overlayState.isPreviewMode()) {
                overlayState.applyInitialPreviewAnchor(client.player.getBlockPos().offset(client.player.getHorizontalFacing()));

                while (rotateKey.wasPressed()) {
                    overlayState.rotateClockwise();
                }
                while (shiftUpKey.wasPressed()) {
                    overlayState.shiftVertical(1);
                }
                while (shiftDownKey.wasPressed()) {
                    overlayState.shiftVertical(-1);
                }
                while (confirmKey.wasPressed()) {
                    overlayState.confirm();
                    client.player.sendMessage(Text.literal("Build confirmed"), true);
                }
                while (cancelKey.wasPressed()) {
                    overlayState.cancel();
                    client.player.sendMessage(Text.literal("Build canceled"), true);
                }
            }

            if (overlayState.isConfirmed()) {
                int removed = overlayState.updateSatisfiedBlocks(pos -> {
                    BlockState state = client.world.getBlockState(pos);
                    return Registries.BLOCK.getId(state.getBlock()).toString();
                });

                if (removed > 0 && overlayState.remainingCount() == 0) {
                    client.player.sendMessage(Text.literal("Build complete"), true);
                    overlayState.cancel();
                }
            }

            if (chatBuildTestWaitTicksRemaining >= 0) {
                buildTestStepTickCount++;
                if (overlayPlacementRevision > chatBuildTestBaselineRevision) {
                    if (buildTestStepFirstPlacementAtMillis < 0) {
                        buildTestStepFirstPlacementAtMillis = System.currentTimeMillis();
                    }
                    if (overlayPlacementRevision != chatBuildTestObservedRevision) {
                        chatBuildTestObservedRevision = overlayPlacementRevision;
                        chatBuildTestStableTicks = 0;
                    } else {
                        chatBuildTestStableTicks++;
                    }
                    if (latestStatusMessage.startsWith("chat: ") && chatBuildTestStableTicks >= CHAT_BUILD_TEST_SETTLE_TICKS) {
                        chatBuildTestWaitTicksRemaining = -1;
                        buildTestStepReadyForCaptureAtMillis = System.currentTimeMillis();
                        buildTestCaptureTicksRemaining = 5;
                    }
                } else if (chatBuildTestWaitTicksRemaining == 0) {
                    chatBuildTestWaitTicksRemaining = -1;
                    latestStatusMessage = "CHAT_BUILD_TEST_TIMEOUT: no blocks placed";
                    if (client.player != null) {
                        client.player.sendMessage(Text.literal(latestStatusMessage), true);
                    }
                } else {
                    chatBuildTestWaitTicksRemaining--;
                }
            }

            if (buildTestCaptureTicksRemaining >= 0) {
                buildTestStepTickCount++;
                buildTestCaptureTicksRemaining--;
                if (buildTestCaptureTicksRemaining == 0) {
                    captureBuildTestArtifacts(client);
                }
            }
        });
    }

    private void registerHudRenderer() {
        HudElementRegistry.attachElementBefore(
                VanillaHudElements.CHAT,
                HUD_LAYER_ID,
                (drawContext, tickCounter) -> renderHud(drawContext)
        );
    }

    private void openHudInput(String prefill) {
        hudChatState.openInput(prefill);
    }

    private void submitChatFromHud(String message) {
        if (message.startsWith("/plan ")) {
            String planPrompt = message.substring("/plan ".length()).trim();
            if (!planPrompt.isEmpty()) {
                commandController.submitPlan(planPrompt);
            }
            return;
        }
        if (message.startsWith("/imagine ")) {
            String imaginePrompt = message.substring("/imagine ".length()).trim();
            if (!imaginePrompt.isEmpty()) {
                commandController.submitImagine(imaginePrompt);
            }
            return;
        }
        if (message.startsWith("/search ")) {
            String searchQuery = message.substring("/search ".length()).trim();
            if (!searchQuery.isEmpty()) {
                commandController.submitSearch(searchQuery);
            }
            return;
        }
        commandController.submitChat(message);
    }

    private String hudStatusLabel() {
        if (activeToolStatus.isBlank()) {
            if (assistantStreaming) {
                return "thinking...";
            }
            return "";
        }
        return activeToolStatus;
    }

    private void registerHudInputCallbacks() {
        MinecraftClient client = MinecraftClient.getInstance();
        client.execute(() -> {
            long windowHandle = client.getWindow().getHandle();
            previousKeyCallback = GLFW.glfwSetKeyCallback(windowHandle, (window, key, scancode, action, mods) -> {
                if (HudInputCapture.shouldConsumeKey(hudChatState.mode(), client.currentScreen != null, action)) {
                    client.execute(() -> onHudKeyInput(client, key, action));
                    return;
                }
                GLFWKeyCallbackI chained = previousKeyCallback;
                if (chained != null) {
                    chained.invoke(window, key, scancode, action, mods);
                }
            });
            previousCharCallback = GLFW.glfwSetCharCallback(windowHandle, (window, codepoint) -> {
                if (HudInputCapture.shouldConsumeChar(hudChatState.mode(), client.currentScreen != null, codepoint)) {
                    client.execute(() -> onHudCharInput(client, codepoint));
                    return;
                }
                GLFWCharCallbackI chained = previousCharCallback;
                if (chained != null) {
                    chained.invoke(window, codepoint);
                }
            });
        });
    }

    private void onHudKeyInput(MinecraftClient client, int key, int action) {
        if (hudChatState.mode() != HudChatState.Mode.INPUT) {
            return;
        }
        if (client.currentScreen != null) {
            return;
        }
        if (action != GLFW.GLFW_PRESS && action != GLFW.GLFW_REPEAT) {
            return;
        }

        switch (key) {
            case GLFW.GLFW_KEY_ESCAPE -> hudChatState.cancelInput();
            case GLFW.GLFW_KEY_ENTER, GLFW.GLFW_KEY_KP_ENTER -> {
                String message = hudChatState.submit();
                if (!message.isEmpty()) {
                    submitChatFromHud(message);
                }
            }
            case GLFW.GLFW_KEY_BACKSPACE -> hudChatState.backspace();
            case GLFW.GLFW_KEY_LEFT -> hudChatState.moveLeft();
            case GLFW.GLFW_KEY_RIGHT -> hudChatState.moveRight();
            default -> {
            }
        }
    }

    private void onHudCharInput(MinecraftClient client, int codepoint) {
        if (hudChatState.mode() != HudChatState.Mode.INPUT) {
            return;
        }
        if (client.currentScreen != null) {
            return;
        }
        if (!Character.isValidCodePoint(codepoint) || Character.isISOControl(codepoint)) {
            return;
        }
        hudChatState.insert((char) codepoint);
    }

    private HudRenderSnapshot buildHudRenderSnapshot(MinecraftClient client) {
        int screenWidth = client.getWindow().getScaledWidth();
        int screenHeight = client.getWindow().getScaledHeight();
        String status = hudStatusLabel();
        String header = status.isBlank() ? "BrowseCraft" : status;
        if (client.options.hudHidden || hudChatState.mode() == HudChatState.Mode.HIDDEN) {
            return new HudRenderSnapshot(
                    hudChatState.mode(),
                    false,
                    screenWidth,
                    screenHeight,
                    header,
                    activeToolStatus,
                    assistantStreaming,
                    null,
                    null,
                    List.of(),
                    0,
                    hudChatState.inputText(),
                    hudChatState.cursor(),
                    0
            );
        }

        int panelWidth = Math.min(HUD_PANEL_WIDTH, screenWidth - (HUD_MARGIN * 2));
        int inputOffset = hudChatState.mode() == HudChatState.Mode.INPUT ? HUD_INPUT_HEIGHT + HUD_MARGIN : 0;
        int maxPanelHeight = screenHeight - (HUD_MARGIN * 2) - inputOffset;
        int panelHeight = Math.min(HUD_PANEL_HEIGHT, Math.max(80, maxPanelHeight));
        int left = screenWidth - panelWidth - HUD_MARGIN;
        int top = HUD_MARGIN;
        int right = left + panelWidth;
        int bottom = top + panelHeight;
        int textWidth = panelWidth - (HUD_INSET * 2);
        int messagesTop = top + HUD_HEADER_HEIGHT;
        int messagesBottom = bottom - HUD_INSET;
        List<RenderedLine> lines = buildRenderedLines(chatHistory, client.textRenderer, textWidth);
        int lineHeight = client.textRenderer.fontHeight + 2;
        int maxVisible = Math.max(1, (messagesBottom - messagesTop) / lineHeight);
        int visibleStartIndex = Math.max(0, lines.size() - maxVisible);
        HudBounds panelBounds = new HudBounds(left, top, right, bottom);
        HudBounds inputBounds = null;
        int cursorX = 0;

        if (hudChatState.mode() == HudChatState.Mode.INPUT) {
            inputBounds = inputBarBounds(screenWidth, screenHeight);
            int cursorIndex = Math.max(0, Math.min(hudChatState.cursor(), hudChatState.inputText().length()));
            cursorX = inputBounds.left() + HUD_INSET
                    + client.textRenderer.getWidth("> " + hudChatState.inputText().substring(0, cursorIndex));
        }

        return new HudRenderSnapshot(
                hudChatState.mode(),
                true,
                screenWidth,
                screenHeight,
                header,
                activeToolStatus,
                assistantStreaming,
                panelBounds,
                inputBounds,
                lines,
                visibleStartIndex,
                hudChatState.inputText(),
                hudChatState.cursor(),
                cursorX
        );
    }

    private void renderHud(DrawContext context) {
        MinecraftClient client = MinecraftClient.getInstance();
        HudRenderSnapshot snapshot = buildHudRenderSnapshot(client);
        latestHudRenderSnapshot = snapshot;
        if (!snapshot.visible()) {
            return;
        }

        HudBounds panelBounds = snapshot.panelBounds();
        int lineHeight = client.textRenderer.fontHeight + 2;
        int messagesTop = panelBounds.top() + HUD_HEADER_HEIGHT;
        int y = messagesTop;
        context.fill(panelBounds.left(), panelBounds.top(), panelBounds.right(), panelBounds.bottom(), 0xB0101010);
        context.drawText(client.textRenderer, snapshot.header(), panelBounds.left() + HUD_INSET, panelBounds.top() + 6, 0xFFE8E8A0, false);

        for (int index = snapshot.visibleStartIndex(); index < snapshot.wrappedLines().size(); index++) {
            RenderedLine line = snapshot.wrappedLines().get(index);
            context.drawText(client.textRenderer, line.orderedText(), panelBounds.left() + HUD_INSET, y, line.color(), false);
            y += lineHeight;
        }

        if (hudChatState.mode() == HudChatState.Mode.INPUT) {
            renderInputBar(context, client.textRenderer, snapshot);
        }
    }

    private void renderInputBar(DrawContext context, TextRenderer textRenderer, HudRenderSnapshot snapshot) {
        HudBounds inputBounds = snapshot.inputBounds();
        context.fill(inputBounds.left(), inputBounds.top(), inputBounds.right(), inputBounds.bottom(), 0xC0202020);
        String prompt = "> " + snapshot.inputText();
        context.drawText(textRenderer, prompt, inputBounds.left() + HUD_INSET, inputBounds.top() + 8, 0xFFFFFFFF, false);

        if ((System.currentTimeMillis() / 500) % 2 == 0) {
            context.fill(
                    snapshot.cursorX(),
                    inputBounds.top() + 6,
                    snapshot.cursorX() + 1,
                    inputBounds.bottom() - 6,
                    0xFFFFFFFF
            );
        }
    }

    private static List<RenderedLine> buildRenderedLines(List<ChatMessage> messages, TextRenderer textRenderer, int width) {
        List<RenderedLine> lines = new ArrayList<>();
        for (ChatMessage message : messages) {
            int color = message.role() == ChatRole.USER ? 0xFF7FD7FF : 0xFFF0F0F0;
            String prefix = message.role() == ChatRole.USER ? "You: " : "AI: ";
            Text text = Text.literal(prefix + message.text());
            List<OrderedText> wrapped = textRenderer.wrapLines(text, width);
            if (wrapped.isEmpty()) {
                OrderedText emptyLine = Text.literal(prefix).asOrderedText();
                lines.add(new RenderedLine(orderedTextToString(emptyLine), emptyLine, color));
                continue;
            }
            for (OrderedText wrappedLine : wrapped) {
                lines.add(new RenderedLine(orderedTextToString(wrappedLine), wrappedLine, color));
            }
        }
        return lines;
    }

    private static HudBounds inputBarBounds(int screenWidth, int screenHeight) {
        int left = HUD_MARGIN;
        int right = screenWidth - HUD_MARGIN;
        int bottom = screenHeight - HUD_MARGIN;
        int top = bottom - HUD_INPUT_HEIGHT;
        return new HudBounds(left, top, right, bottom);
    }

    private static String orderedTextToString(OrderedText text) {
        StringBuilder builder = new StringBuilder();
        text.accept((index, style, codePoint) -> {
            builder.appendCodePoint(codePoint);
            return true;
        });
        return builder.toString();
    }

    private void handleChatUserMessage(String message) {
        hudChatState.ensureHudVisible();
        chatHistory.add(new ChatMessage(ChatRole.USER, message));
        assistantStreaming = true;
    }

    private void handleChatAssistantDelta(String delta) {
        if (delta.isEmpty()) {
            return;
        }
        int lastIndex = chatHistory.size() - 1;
        if (lastIndex < 0 || chatHistory.get(lastIndex).role() != ChatRole.ASSISTANT || !assistantStreaming) {
            chatHistory.add(new ChatMessage(ChatRole.ASSISTANT, delta));
        } else {
            ChatMessage last = chatHistory.get(lastIndex);
            chatHistory.set(lastIndex, new ChatMessage(last.role(), last.text() + delta));
        }
        hudChatState.ensureHudVisible();
        assistantStreaming = true;
    }

    private void handleChatAssistantMessage(String message) {
        int lastIndex = chatHistory.size() - 1;
        if (lastIndex >= 0 && chatHistory.get(lastIndex).role() == ChatRole.ASSISTANT) {
            chatHistory.set(lastIndex, new ChatMessage(ChatRole.ASSISTANT, message));
        } else {
            chatHistory.add(new ChatMessage(ChatRole.ASSISTANT, message));
        }
        hudChatState.ensureHudVisible();
        assistantStreaming = false;
        activeToolStatus = "";
    }

    private void handleToolStatus(String status) {
        activeToolStatus = status;
        hudChatState.ensureHudVisible();
    }

    private void runBuildTestCommand(MinecraftClient client) {
        if (client.player == null) {
            return;
        }

        latestBuildTestJsonPath = null;
        latestBuildTestScreenshotPath = null;
        beginBuildTestTrace("build-test", null);
        chatBuildTestWaitTicksRemaining = -1;
        BuildPlan testPlan = createBuildTestPlan();
        overlayState.setPlan(testPlan);
        overlayState.setRotationQuarterTurns(rotationForFacing(client.player.getHorizontalFacing()));
        buildTestCaptureTicksRemaining = 5;
        client.player.sendMessage(Text.literal("Loaded /build-test plan"), true);
    }

    private void runHudTestCommand(MinecraftClient client) {
        if (client.player == null) {
            return;
        }

        if (hudCaptureSession != null) {
            restoreHudDebugState(hudCaptureSession.previousState);
        }

        latestHudTestJsonPath = null;
        HudDebugState previousState = snapshotHudDebugState();
        hudCaptureSession = new HudCaptureSession(
                Instant.now().toEpochMilli(),
                previousState,
                List.of(
                        new HudCaptureTarget(HudChatState.Mode.HIDDEN, "hidden"),
                        new HudCaptureTarget(HudChatState.Mode.HUD, "hud"),
                        new HudCaptureTarget(HudChatState.Mode.INPUT, "input")
                )
        );
        prepareHudCaptureState(hudCaptureSession.currentTarget());
        hudCaptureSession.settleTicksRemaining = HUD_TEST_SETTLE_TICKS;
        client.player.sendMessage(Text.literal("Running /hud-test captures"), true);
    }

    private HudDebugState snapshotHudDebugState() {
        return new HudDebugState(
                List.copyOf(chatHistory),
                activeToolStatus,
                assistantStreaming,
                hudChatState.snapshot()
        );
    }

    private void restoreHudDebugState(HudDebugState state) {
        chatHistory.clear();
        chatHistory.addAll(state.chatHistory());
        activeToolStatus = state.activeToolStatus();
        assistantStreaming = state.assistantStreaming();
        hudChatState.restore(state.hudState());
    }

    private void prepareHudCaptureState(HudCaptureTarget target) {
        chatHistory.clear();
        chatHistory.add(new ChatMessage(ChatRole.USER, "build a lantern arch over the path"));
        chatHistory.add(new ChatMessage(ChatRole.ASSISTANT, "Placing the main arch supports with spruce logs."));
        chatHistory.add(new ChatMessage(ChatRole.ASSISTANT, "Next step: hang lanterns under the center span."));
        activeToolStatus = "Step 2/3: placing supports...";
        assistantStreaming = false;

        switch (target.mode()) {
            case HIDDEN -> hudChatState.restore(new HudChatState.Snapshot(HudChatState.Mode.HIDDEN, "", 0));
            case HUD -> hudChatState.restore(new HudChatState.Snapshot(HudChatState.Mode.HUD, "", 0));
            case INPUT -> {
                String input = "/chat add lanterns to the arch";
                hudChatState.restore(new HudChatState.Snapshot(HudChatState.Mode.INPUT, input, input.length()));
            }
        }
    }

    private void runChatBuildTestCommand(MinecraftClient client, String message, boolean resetOverlay) {
        if (client.player == null) {
            return;
        }

        latestBuildTestJsonPath = null;
        latestBuildTestScreenshotPath = null;
        beginBuildTestTrace("chat", message);
        buildTestCaptureTicksRemaining = -1;
        chatBuildTestWaitTicksRemaining = CHAT_BUILD_TEST_TIMEOUT_TICKS;
        chatBuildTestBaselineRevision = overlayPlacementRevision;
        chatBuildTestObservedRevision = overlayPlacementRevision;
        chatBuildTestStableTicks = 0;

        if (resetOverlay) {
            overlayState.cancel();
            hasUndoState = false;
            undoClearsOverlay = false;
            undoBlueprintState = null;
            undoWorldPlacements = null;
        }

        commandController.submitChat(message);
    }

    private BuildPlan createBuildTestPlan() {
        List<BuildPlacement> placements = new ArrayList<>();

        for (int x = 0; x < 5; x++) {
            for (int z = 0; z < 5; z++) {
                placements.add(new BuildPlacement(x, 0, z, "minecraft:stone", Map.of()));
            }
        }

        for (int y = 1; y <= 2; y++) {
            for (int x = 0; x < 5; x++) {
                for (int z = 0; z < 5; z++) {
                    boolean onBoundary = x == 0 || x == 4 || z == 0 || z == 4;
                    if (!onBoundary) {
                        continue;
                    }
                    if (z == 0 && x == 2) {
                        continue;
                    }

                    String blockId = "minecraft:cobblestone";
                    if (y == 1 && ((x == 0 && z == 2) || (x == 4 && z == 2))) {
                        blockId = "minecraft:glass_pane";
                    }
                    placements.add(new BuildPlacement(x, y, z, blockId, Map.of()));
                }
            }
        }

        for (int x = 0; x < 5; x++) {
            for (int z = 0; z < 5; z++) {
                placements.add(new BuildPlacement(x, 3, z, "minecraft:oak_planks", Map.of()));
            }
        }

        return new BuildPlan(placements.size(), List.copyOf(placements));
    }

    private int commandSaveBlueprint(MinecraftClient client, String name) {
        if (client.player == null) {
            throw new IllegalStateException("blueprints save requires an active player");
        }
        if (!overlayState.hasPlan()) {
            client.player.sendMessage(Text.literal("No active overlay to save"), false);
            return 0;
        }

        BlueprintStore store = blueprintStore(client);
        try {
            store.save(name, overlayState.blueprintState());
        } catch (IOException error) {
            throw new RuntimeException("Failed to save blueprint '" + name + "'", error);
        }

        client.player.sendMessage(Text.literal("Saved blueprint " + name), false);
        return 1;
    }

    private int commandLoadBlueprint(MinecraftClient client, String name) {
        if (client.player == null) {
            throw new IllegalStateException("blueprints load requires an active player");
        }

        BlueprintStore store = blueprintStore(client);
        OverlayState.BlueprintState blueprintState;
        try {
            blueprintState = store.load(name);
        } catch (IOException error) {
            throw new RuntimeException("Failed to load blueprint '" + name + "'", error);
        }
        overlayState.loadBlueprint(blueprintState);

        client.player.sendMessage(Text.literal("Loaded blueprint " + name + " (" + blueprintState.plan().totalBlocks() + " blocks)"), false);
        return 1;
    }

    private int commandListBlueprints(MinecraftClient client) {
        if (client.player == null) {
            throw new IllegalStateException("blueprints list requires an active player");
        }

        BlueprintStore store = blueprintStore(client);
        List<String> names;
        try {
            names = store.list();
        } catch (IOException error) {
            throw new RuntimeException("Failed to list blueprints", error);
        }

        if (names.isEmpty()) {
            client.player.sendMessage(Text.literal("No saved blueprints"), false);
            return 1;
        }

        client.player.sendMessage(Text.literal(String.join(", ", names)), false);
        return 1;
    }

    private int commandMaterials(MinecraftClient client) {
        if (client.player == null) {
            throw new IllegalStateException("materials requires an active player");
        }

        List<OverlayState.TransformedPlacement> placements = overlayState.transformedPlacements();
        if (placements.isEmpty()) {
            client.player.sendMessage(Text.literal("No active overlay"), false);
            return 0;
        }

        List<MaterialSummary.MaterialCount> materials = MaterialSummary.aggregate(placements);
        List<String> materialText = new ArrayList<>(materials.size());
        for (MaterialSummary.MaterialCount material : materials) {
            materialText.add(blockDisplayName(material.blockId()) + " x " + material.count());
        }
        client.player.sendMessage(Text.literal(String.join(", ", materialText)), false);

        if (overlayState.isConfirmed()) {
            Map<String, Integer> inventoryCounts = inventoryBlockCounts(client.player.getInventory());
            List<MaterialSummary.MaterialDelta> deltas = MaterialSummary.compareWithInventory(materials, inventoryCounts);
            List<String> deltaText = new ArrayList<>(deltas.size());
            for (MaterialSummary.MaterialDelta delta : deltas) {
                deltaText.add(blockDisplayName(delta.blockId())
                        + " need " + delta.needed()
                        + ", inv " + delta.inInventory()
                        + ", missing " + delta.missing());
            }
            client.player.sendMessage(Text.literal(String.join(" | ", deltaText)), false);
        }

        return 1;
    }

    private Path worldSaveDir(MinecraftClient client) {
        if (client.getServer() == null) {
            throw new IllegalStateException("blueprints commands require a local singleplayer world");
        }
        return client.getServer().getSavePath(WorldSavePath.ROOT);
    }

    private BlueprintStore blueprintStore(MinecraftClient client) {
        return new BlueprintStore(BlueprintStore.defaultDirectory(worldSaveDir(client)));
    }

    private void refreshAvailableBlockTypes(MinecraftClient client) {
        if (client.player == null) {
            return;
        }
        Map<String, Integer> inventoryCounts = inventoryBlockCounts(client.player.getInventory());
        ghostRenderer.setAvailableBlockTypes(inventoryCounts.keySet());
    }

    private Map<String, Integer> inventoryBlockCounts(PlayerInventory inventory) {
        Map<String, Integer> counts = new HashMap<>();
        for (int slot = 0; slot < inventory.size(); slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack.isEmpty()) {
                continue;
            }
            if (!(stack.getItem() instanceof BlockItem blockItem)) {
                continue;
            }
            String blockId = Registries.BLOCK.getId(blockItem.getBlock()).toString();
            counts.merge(blockId, stack.getCount(), Integer::sum);
        }
        return counts;
    }

    private String blockDisplayName(String blockId) {
        Identifier identifier = Identifier.of(blockId);
        return Registries.BLOCK.get(identifier).getName().getString();
    }

    private int rotationForFacing(Direction facing) {
        return switch (facing) {
            case NORTH -> 0;
            case EAST -> 1;
            case SOUTH -> 2;
            case WEST -> 3;
            default -> 0;
        };
    }

    private void tickHudCaptureSession(MinecraftClient client) {
        HudCaptureSession session = hudCaptureSession;
        if (session == null) {
            return;
        }
        if (session.settleTicksRemaining > 0) {
            session.settleTicksRemaining--;
            return;
        }

        try {
            captureHudTestArtifact(client, session, session.currentTarget());
            session.captureIndex++;
            if (session.captureIndex >= session.targets.size()) {
                Path manifestPath = writeHudTestManifest(client, session);
                latestHudTestJsonPath = manifestPath;
                restoreHudDebugState(session.previousState);
                hudCaptureSession = null;
                if (client.player != null) {
                    client.player.sendMessage(Text.literal("Saved /hud-test artifacts to " + manifestPath), true);
                }
                if (Boolean.getBoolean(HUD_TEST_EXIT_ON_COMPLETE_PROPERTY)) {
                    client.scheduleStop();
                }
                return;
            }

            prepareHudCaptureState(session.currentTarget());
            session.settleTicksRemaining = HUD_TEST_SETTLE_TICKS;
        } catch (IOException error) {
            throw new RuntimeException("Failed to capture /hud-test artifacts", error);
        }
    }

    private void captureHudTestArtifact(MinecraftClient client, HudCaptureSession session, HudCaptureTarget target) throws IOException {
        long capturedAtMillis = System.currentTimeMillis();
        Path runDir = client.runDirectory.toPath();
        Path hudTestDir = runDir.resolve("browsecraft").resolve("hud-test");
        Files.createDirectories(hudTestDir);
        String screenshotName = "browsecraft-hud-test-" + session.sessionTimestamp + "-" + target.label() + ".png";
        Path screenshotPath = runDir.resolve("screenshots").resolve(screenshotName).toAbsolutePath();

        ScreenshotRecorder.saveScreenshot(
                client.runDirectory,
                screenshotName,
                client.getFramebuffer(),
                1,
                message -> {}
        );

        session.captures.add(hudCaptureToJson(target, screenshotPath, capturedAtMillis));
    }

    private Path writeHudTestManifest(MinecraftClient client, HudCaptureSession session) throws IOException {
        Path hudTestDir = client.runDirectory.toPath().resolve("browsecraft").resolve("hud-test");
        Files.createDirectories(hudTestDir);

        JsonObject root = new JsonObject();
        root.addProperty("trigger", "hud-test");
        root.addProperty("session_timestamp_ms", session.sessionTimestamp);
        root.addProperty("run_directory", client.runDirectory.toPath().toAbsolutePath().toString());
        root.addProperty("latest_status_message", latestStatusMessage);
        root.addProperty("auto_started", hudTestStartupTriggered);
        root.addProperty("auto_exit_enabled", Boolean.getBoolean(HUD_TEST_EXIT_ON_COMPLETE_PROPERTY));

        JsonArray captures = new JsonArray();
        for (JsonObject capture : session.captures) {
            captures.add(capture.deepCopy());
        }
        root.add("captures", captures);

        String payload = GSON.toJson(root);
        Path manifestPath = hudTestDir.resolve("hud-test-" + session.sessionTimestamp + ".json");
        Path latestManifestPath = hudTestDir.resolve("latest.json");
        Files.writeString(manifestPath, payload);
        Files.writeString(latestManifestPath, payload);
        return manifestPath;
    }

    private JsonObject hudCaptureToJson(HudCaptureTarget target, Path screenshotPath, long capturedAtMillis) {
        HudRenderSnapshot snapshot = latestHudRenderSnapshot;
        JsonObject capture = new JsonObject();
        capture.addProperty("label", target.label());
        capture.addProperty("mode", target.mode().name());
        capture.addProperty("captured_at_ms", capturedAtMillis);
        capture.addProperty("screenshot_path", screenshotPath.toString());
        capture.addProperty("visible", snapshot.visible());
        capture.addProperty("screen_width", snapshot.screenWidth());
        capture.addProperty("screen_height", snapshot.screenHeight());
        capture.addProperty("header", snapshot.header());
        capture.addProperty("active_status_text", snapshot.activeToolStatus());
        capture.addProperty("assistant_streaming", snapshot.assistantStreaming());
        capture.addProperty("input_text", snapshot.inputText());
        capture.addProperty("cursor_index", snapshot.cursorIndex());
        capture.addProperty("cursor_x", snapshot.cursorX());
        capture.addProperty("visible_start_index", snapshot.visibleStartIndex());
        if (snapshot.panelBounds() != null) {
            capture.add("panel_bounds", hudBoundsToJson(snapshot.panelBounds()));
        }
        if (snapshot.inputBounds() != null) {
            capture.add("input_bounds", hudBoundsToJson(snapshot.inputBounds()));
        }

        JsonArray wrappedLines = new JsonArray();
        for (RenderedLine line : snapshot.wrappedLines()) {
            wrappedLines.add(line.plainText());
        }
        capture.add("wrapped_lines", wrappedLines);

        JsonArray visibleWrappedLines = new JsonArray();
        for (int index = snapshot.visibleStartIndex(); index < snapshot.wrappedLines().size(); index++) {
            visibleWrappedLines.add(snapshot.wrappedLines().get(index).plainText());
        }
        capture.add("visible_wrapped_lines", visibleWrappedLines);
        return capture;
    }

    private JsonObject hudBoundsToJson(HudBounds bounds) {
        JsonObject json = new JsonObject();
        json.addProperty("left", bounds.left());
        json.addProperty("top", bounds.top());
        json.addProperty("right", bounds.right());
        json.addProperty("bottom", bounds.bottom());
        json.addProperty("width", bounds.right() - bounds.left());
        json.addProperty("height", bounds.bottom() - bounds.top());
        return json;
    }

    private void captureBuildTestArtifacts(MinecraftClient client) {
        try {
            OverlaySnapshot snapshot = overlayState.snapshot();
            GhostRenderer.RenderFrameSnapshot renderSnapshot = ghostRenderer.latestFrameSnapshot();
            long capturedAtMillis = System.currentTimeMillis();
            long artifactTimestamp = Instant.now().toEpochMilli();

            Path runDir = client.runDirectory.toPath();
            Path buildTestDir = runDir.resolve("browsecraft").resolve("build-test");
            Path screenshotsDir = runDir.resolve("screenshots");
            Files.createDirectories(buildTestDir);
            Files.createDirectories(screenshotsDir);

            Path jsonPath = buildTestDir.resolve("ghost-state-" + artifactTimestamp + ".json");
            Path latestJsonPath = buildTestDir.resolve("ghost-state.json");
            Path screenshotPath = screenshotsDir.resolve("browsecraft-build-test-" + artifactTimestamp + ".png");

            JsonObject root = new JsonObject();
            JsonObject test = new JsonObject();
            test.addProperty("trigger", buildTestTrigger);
            if (buildTestPrompt != null) {
                test.addProperty("prompt", buildTestPrompt);
            }
            root.add("test", test);
            root.add("step", buildTestStepMetadata(capturedAtMillis));
            root.add("overlay", overlayToJson(snapshot));
            root.add("validation", validationToJson(snapshot));
            JsonArray toolCalls = new JsonArray();
            synchronized (buildTestToolCalls) {
                for (JsonObject toolCall : buildTestToolCalls) {
                    toolCalls.add(toolCall.deepCopy());
                }
            }
            root.add("tool_calls", toolCalls);
            root.add("status_timeline", buildTestStatusTimelineJson());
            JsonObject backend = new JsonObject();
            backend.addProperty("latest_status_message", latestStatusMessage);
            root.add("backend", backend);

            JsonObject renderStatus = new JsonObject();
            renderStatus.addProperty("mode", renderSnapshot.mode().name());
            renderStatus.addProperty("placement_count", renderSnapshot.placementCount());
            root.add("render_status", renderStatus);
            root.addProperty("render_status_observed", renderSnapshot.mode() != GhostRenderer.RenderMode.NONE);

            if (client.player != null) {
                JsonObject player = new JsonObject();
                player.addProperty("x", client.player.getX());
                player.addProperty("y", client.player.getY());
                player.addProperty("z", client.player.getZ());
                player.addProperty("facing", client.player.getHorizontalFacing().asString());
                root.add("player", player);
            }

            String payload = GSON.toJson(root);
            Files.writeString(jsonPath, payload);
            Files.writeString(latestJsonPath, payload);
            ScreenshotRecorder.saveScreenshot(
                    client.runDirectory,
                    screenshotPath.getFileName().toString(),
                    client.getFramebuffer(),
                    1,
                    message -> {}
            );

            latestBuildTestJsonPath = jsonPath;
            latestBuildTestScreenshotPath = screenshotPath;
        } catch (IOException error) {
            if (client.player != null) {
                client.player.sendMessage(Text.literal("Failed to write /build-test artifacts: " + error.getMessage()), true);
            }
        }
    }

    private JsonObject overlayToJson(OverlaySnapshot snapshot) {
        JsonObject overlay = new JsonObject();
        overlay.addProperty("has_plan", snapshot.hasPlan());
        overlay.addProperty("preview_mode", snapshot.previewMode());
        overlay.addProperty("confirmed", snapshot.confirmed());
        overlay.addProperty("rotation_quarter_turns", snapshot.rotationQuarterTurns());
        overlay.addProperty("remaining_count", snapshot.remainingCount());

        JsonObject anchor = new JsonObject();
        anchor.addProperty("x", snapshot.anchor().getX());
        anchor.addProperty("y", snapshot.anchor().getY());
        anchor.addProperty("z", snapshot.anchor().getZ());
        overlay.add("anchor", anchor);

        JsonArray placements = new JsonArray();
        for (OverlayState.TransformedPlacement transformed : snapshot.transformedPlacements()) {
            JsonObject placement = new JsonObject();
            placement.addProperty("x", transformed.pos().getX());
            placement.addProperty("y", transformed.pos().getY());
            placement.addProperty("z", transformed.pos().getZ());
            placement.addProperty("block_id", transformed.blockId());
            placements.add(placement);
        }
        overlay.add("transformed_placements", placements);

        return overlay;
    }

    private JsonObject validationToJson(OverlaySnapshot snapshot) {
        JsonObject validation = new JsonObject();
        boolean passed = snapshot.remainingCount() == snapshot.transformedPlacements().size() && snapshot.hasPlan();
        validation.addProperty("passed", passed);
        validation.addProperty("remaining_count", snapshot.remainingCount());
        validation.addProperty("transformed_count", snapshot.transformedPlacements().size());
        return validation;
    }

    private CompletableFuture<JsonObject> handleToolRequest(String tool, JsonObject params) {
        CompletableFuture<JsonObject> future = new CompletableFuture<>();
        MinecraftClient client = MinecraftClient.getInstance();
        client.execute(() -> {
            try {
                future.complete(dispatchToolRequest(client, tool, params));
            } catch (Throwable error) {
                future.completeExceptionally(error);
            }
        });
        return future;
    }

    private JsonObject dispatchToolRequest(MinecraftClient client, String tool, JsonObject params) {
        long startedAtMillis = System.currentTimeMillis();
        try {
            JsonObject result = switch (tool) {
                case "player_position" -> toolPlayerPosition(client);
                case "player_inventory" -> toolPlayerInventory(client);
                case "inspect_area" -> toolInspectArea(client, params);
                case "place_blocks" -> toolPlaceBlocks(client, params);
                case "fill_region" -> toolFillRegion(client, params);
                case "set_plan" -> toolSetPlan(client, params);
                case "undo_last" -> toolUndoLast(client);
                case "get_active_overlay" -> activeOverlaySummary(overlayState.snapshot());
                case "modify_overlay" -> toolModifyOverlay(params);
                case "get_blueprints" -> toolGetBlueprints(client);
                case "save_blueprint" -> toolSaveBlueprint(client, params);
                case "load_blueprint" -> toolLoadBlueprint(client, params);
                default -> throw new IllegalArgumentException("Unsupported tool: " + tool);
            };
            long completedAtMillis = System.currentTimeMillis();
            recordBuildTestToolCall(tool, params, result, null, startedAtMillis, completedAtMillis);
            return result;
        } catch (RuntimeException | Error error) {
            long completedAtMillis = System.currentTimeMillis();
            recordBuildTestToolCall(tool, params, null, error, startedAtMillis, completedAtMillis);
            throw error;
        }
    }

    private JsonObject toolPlayerPosition(MinecraftClient client) {
        if (client.player == null) {
            throw new IllegalStateException("player_position requires an active player");
        }
        if (client.world == null) {
            throw new IllegalStateException("player_position requires an active world");
        }

        JsonObject position = new JsonObject();
        position.addProperty("x", client.player.getX());
        position.addProperty("y", client.player.getY());
        position.addProperty("z", client.player.getZ());
        position.addProperty("yaw", client.player.getYaw());
        position.addProperty("pitch", client.player.getPitch());
        position.addProperty("block_x", client.player.getBlockX());
        position.addProperty("block_y", client.player.getBlockY());
        position.addProperty("block_z", client.player.getBlockZ());
        position.addProperty("ground_y", findGroundY(client.world, client.player.getBlockPos()));
        position.addProperty("facing", client.player.getHorizontalFacing().asString());
        position.addProperty("dimension", client.world.getRegistryKey().getValue().toString());
        return position;
    }

    private JsonObject toolPlayerInventory(MinecraftClient client) {
        if (client.player == null) {
            throw new IllegalStateException("player_inventory requires an active player");
        }

        PlayerInventory inventory = client.player.getInventory();
        JsonArray items = new JsonArray();
        int totalItemCount = 0;

        for (int slot = 0; slot < inventory.size(); slot++) {
            ItemStack stack = inventory.getStack(slot);
            if (stack.isEmpty()) {
                continue;
            }

            JsonObject item = new JsonObject();
            item.addProperty("slot", slot);
            item.addProperty("item_id", Registries.ITEM.getId(stack.getItem()).toString());
            item.addProperty("count", stack.getCount());
            items.add(item);
            totalItemCount += stack.getCount();
        }

        JsonObject result = new JsonObject();
        result.addProperty("selected_slot", inventory.getSelectedSlot());
        result.addProperty("filled_slots", items.size());
        result.addProperty("total_item_count", totalItemCount);
        result.add("items", items);
        return result;
    }

    private JsonObject toolInspectArea(MinecraftClient client, JsonObject params) {
        if (client.player == null) {
            throw new IllegalStateException("inspect_area requires an active player");
        }
        if (client.world == null) {
            throw new IllegalStateException("inspect_area requires an active world");
        }

        BlockPos center = requiredBlockPos(params, "center");
        int requestedRadius = requiredInt(params, "radius");
        boolean detailed = params.has("detailed") && requiredBoolean(params, "detailed");
        boolean filterTerrain = !params.has("filter_terrain") || requiredBoolean(params, "filter_terrain");
        int maxRadius = detailed ? 6 : 12;
        int radius = Math.min(maxRadius, Math.max(0, requestedRadius));
        ClientWorld world = client.world;
        Map<String, Integer> blockCounts = new TreeMap<>();
        JsonArray nonAirBlocks = detailed ? new JsonArray() : null;
        int sampledBlocks = 0;

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    BlockPos pos = center.add(dx, dy, dz);
                    String blockId = Registries.BLOCK.getId(world.getBlockState(pos).getBlock()).toString();
                    blockCounts.merge(blockId, 1, Integer::sum);
                    if (
                            detailed
                                    && !"minecraft:air".equals(blockId)
                                    && !(filterTerrain && isTerrainBlock(blockId, pos.getY()))
                    ) {
                        JsonObject block = new JsonObject();
                        block.addProperty("x", pos.getX());
                        block.addProperty("y", pos.getY());
                        block.addProperty("z", pos.getZ());
                        block.addProperty("block_id", blockId);
                        nonAirBlocks.add(block);
                    }
                    sampledBlocks++;
                }
            }
        }

        JsonObject countsJson = new JsonObject();
        for (Map.Entry<String, Integer> entry : blockCounts.entrySet()) {
            countsJson.addProperty(entry.getKey(), entry.getValue());
        }

        JsonObject result = new JsonObject();
        result.addProperty("requested_radius", requestedRadius);
        result.addProperty("radius", radius);
        result.addProperty("sampled_blocks", sampledBlocks);
        result.add("center", blockPosToJson(center));
        result.add("block_counts", countsJson);
        result.addProperty("detailed", detailed);
        result.addProperty("filter_terrain", filterTerrain);
        if (detailed) {
            result.add("non_air_blocks", nonAirBlocks);
        }
        return result;
    }

    private JsonObject toolPlaceBlocks(MinecraftClient client, JsonObject params) {
        JsonElement placementsElement = params.get("placements");
        if (placementsElement == null || !placementsElement.isJsonArray()) {
            throw new IllegalArgumentException("Expected array field: placements");
        }
        JsonArray placementsJson = placementsElement.getAsJsonArray();
        if (placementsJson.isEmpty()) {
            throw new IllegalArgumentException("placements must contain at least one block");
        }

        List<AbsolutePlacement> absolutePlacements = new ArrayList<>(placementsJson.size());

        for (JsonElement element : placementsJson) {
            if (!element.isJsonObject()) {
                throw new IllegalArgumentException("placements must contain objects");
            }
            JsonObject placementJson = element.getAsJsonObject();
            int x = requiredInt(placementJson, "x");
            int y = requiredInt(placementJson, "y");
            int z = requiredInt(placementJson, "z");
            String blockId = normalizeBlockId(requiredString(placementJson, "block_id"));

            absolutePlacements.add(new AbsolutePlacement(x, y, z, blockId));
        }

        return applyAbsolutePlacements(client, absolutePlacements);
    }

    private JsonObject toolFillRegion(MinecraftClient client, JsonObject params) {
        BlockPos fromCorner = requiredBlockPos(params, "from_corner");
        BlockPos toCorner = requiredBlockPos(params, "to_corner");
        String blockId = normalizeBlockId(requiredString(params, "block_id"));

        int minX = Math.min(fromCorner.getX(), toCorner.getX());
        int maxX = Math.max(fromCorner.getX(), toCorner.getX());
        int minY = Math.min(fromCorner.getY(), toCorner.getY());
        int maxY = Math.max(fromCorner.getY(), toCorner.getY());
        int minZ = Math.min(fromCorner.getZ(), toCorner.getZ());
        int maxZ = Math.max(fromCorner.getZ(), toCorner.getZ());

        long volume = (long) (maxX - minX + 1) * (long) (maxY - minY + 1) * (long) (maxZ - minZ + 1);
        if (volume > 4096L) {
            throw new IllegalArgumentException("fill_region volume must be <= 4096 blocks");
        }

        JsonObject result = applyFilledRegion(client, minX, minY, minZ, maxX, maxY, maxZ, blockId);
        result.addProperty("fill_region", true);
        result.add("from_corner", blockPosToJson(fromCorner));
        result.add("to_corner", blockPosToJson(toCorner));
        return result;
    }

    private JsonObject toolSetPlan(MinecraftClient client, JsonObject params) {
        JsonElement placementsElement = params.get("placements");
        if (placementsElement == null || !placementsElement.isJsonArray()) {
            throw new IllegalArgumentException("Expected array field: placements");
        }
        JsonArray placementsJson = placementsElement.getAsJsonArray();
        if (placementsJson.isEmpty()) {
            throw new IllegalArgumentException("placements must contain at least one block");
        }

        List<BuildPlacement> relativePlacements = new ArrayList<>(placementsJson.size());
        for (JsonElement element : placementsJson) {
            if (!element.isJsonObject()) {
                throw new IllegalArgumentException("placements must contain objects");
            }
            JsonObject placementJson = element.getAsJsonObject();
            relativePlacements.add(new BuildPlacement(
                    requiredInt(placementJson, "dx"),
                    requiredInt(placementJson, "dy"),
                    requiredInt(placementJson, "dz"),
                    normalizeBlockId(requiredString(placementJson, "block_id")),
                    Map.of()
            ));
        }

        overlayState.setPlan(new BuildPlan(relativePlacements.size(), List.copyOf(relativePlacements)));
        if (client.player != null) {
            overlayState.applyInitialPreviewAnchor(client.player.getBlockPos().offset(client.player.getHorizontalFacing()));
        }
        overlayPlacementRevision++;

        JsonObject result = new JsonObject();
        result.addProperty("loaded_count", relativePlacements.size());
        result.add("overlay", activeOverlaySummary(overlayState.snapshot()));
        return result;
    }

    private JsonObject applyAbsolutePlacements(MinecraftClient client, List<AbsolutePlacement> absolutePlacements) {
        if (absolutePlacements.isEmpty()) {
            throw new IllegalArgumentException("placements must contain at least one block");
        }
        if (client.player == null || client.world == null || client.getNetworkHandler() == null) {
            throw new IllegalStateException("place_blocks requires an active player, world, and network handler");
        }

        if (overlayState.hasPlan()) {
            overlayState.cancel();
        }
        Map<BlockPos, String> targetByPos = new LinkedHashMap<>();
        for (AbsolutePlacement placement : absolutePlacements) {
            targetByPos.put(new BlockPos(placement.x(), placement.y(), placement.z()), placement.blockId());
        }

        int anchorX = Integer.MAX_VALUE;
        int anchorY = Integer.MAX_VALUE;
        int anchorZ = Integer.MAX_VALUE;
        List<PlacementBatchPlanner.Placement> plannerInput = new ArrayList<>(targetByPos.size());
        List<AbsolutePlacement> previousPlacements = new ArrayList<>(targetByPos.size());
        for (Map.Entry<BlockPos, String> entry : targetByPos.entrySet()) {
            BlockPos pos = entry.getKey();
            String previousBlockId = Registries.BLOCK.getId(client.world.getBlockState(pos).getBlock()).toString();
            plannerInput.add(new PlacementBatchPlanner.Placement(pos.getX(), pos.getY(), pos.getZ(), entry.getValue()));
            previousPlacements.add(new AbsolutePlacement(pos.getX(), pos.getY(), pos.getZ(), previousBlockId));
            anchorX = Math.min(anchorX, pos.getX());
            anchorY = Math.min(anchorY, pos.getY());
            anchorZ = Math.min(anchorZ, pos.getZ());
        }

        PlacementBatchPlanner.Plan batchPlan = PlacementBatchPlanner.plan(plannerInput);
        for (PlacementBatchPlanner.Cuboid cuboid : batchPlan.fillCuboids()) {
            sendCommand(client, fillCommand(
                    cuboid.minX(),
                    cuboid.minY(),
                    cuboid.minZ(),
                    cuboid.maxX(),
                    cuboid.maxY(),
                    cuboid.maxZ(),
                    cuboid.blockId()
            ));
        }
        List<String> queuedSetblockCommands = new ArrayList<>(batchPlan.setBlocks().size());
        for (PlacementBatchPlanner.Placement setBlock : batchPlan.setBlocks()) {
            queuedSetblockCommands.add(setblockCommand(setBlock.x(), setBlock.y(), setBlock.z(), setBlock.blockId()));
        }
        for (String command : queuedSetblockCommands) {
            sendCommand(client, command);
        }

        undoWorldPlacements = previousPlacements;
        undoBlueprintState = null;
        undoClearsOverlay = false;
        hasUndoState = true;
        overlayPlacementRevision++;
        BlockPos anchor = new BlockPos(anchorX, anchorY, anchorZ);

        JsonObject result = new JsonObject();
        result.addProperty("placed_count", targetByPos.size());
        result.addProperty("fill_count", batchPlan.fillCuboids().size());
        result.addProperty("setblock_count", batchPlan.setBlocks().size());
        result.add("anchor", blockPosToJson(anchor));
        return result;
    }

    private JsonObject applyFilledRegion(
            MinecraftClient client,
            int minX,
            int minY,
            int minZ,
            int maxX,
            int maxY,
            int maxZ,
            String blockId
    ) {
        if (client.player == null || client.world == null || client.getNetworkHandler() == null) {
            throw new IllegalStateException("fill_region requires an active player, world, and network handler");
        }

        if (overlayState.hasPlan()) {
            overlayState.cancel();
        }

        List<AbsolutePlacement> previousPlacements = new ArrayList<>((maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1));
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                for (int z = minZ; z <= maxZ; z++) {
                    BlockPos pos = new BlockPos(x, y, z);
                    String previousBlockId = Registries.BLOCK.getId(client.world.getBlockState(pos).getBlock()).toString();
                    previousPlacements.add(new AbsolutePlacement(x, y, z, previousBlockId));
                }
            }
        }

        sendCommand(client, fillCommand(minX, minY, minZ, maxX, maxY, maxZ, blockId));
        undoWorldPlacements = previousPlacements;
        undoBlueprintState = null;
        undoClearsOverlay = false;
        hasUndoState = true;
        overlayPlacementRevision++;

        JsonObject result = new JsonObject();
        int placedCount = previousPlacements.size();
        result.addProperty("placed_count", placedCount);
        result.add("anchor", blockPosToJson(new BlockPos(minX, minY, minZ)));
        return result;
    }

    private JsonObject toolUndoLast(MinecraftClient client) {
        if (!hasUndoState) {
            throw new IllegalStateException("No placement batch to undo");
        }

        if (undoWorldPlacements != null) {
            if (client.player == null || client.getNetworkHandler() == null) {
                throw new IllegalStateException("undo_last requires an active player and network handler");
            }
            for (AbsolutePlacement placement : undoWorldPlacements) {
                sendCommand(client, setblockCommand(placement.x(), placement.y(), placement.z(), placement.blockId()));
            }
            undoWorldPlacements = null;
        } else if (undoClearsOverlay) {
            overlayState.cancel();
        } else {
            overlayState.loadBlueprint(undoBlueprintState);
        }

        hasUndoState = false;
        undoClearsOverlay = false;
        undoBlueprintState = null;

        JsonObject result = new JsonObject();
        result.addProperty("undone", true);
        result.add("overlay", activeOverlaySummary(overlayState.snapshot()));
        return result;
    }

    private String fillCommand(int minX, int minY, int minZ, int maxX, int maxY, int maxZ, String blockId) {
        return "fill " + minX + " " + minY + " " + minZ + " " + maxX + " " + maxY + " " + maxZ + " " + blockId + " replace";
    }

    private String setblockCommand(int x, int y, int z, String blockId) {
        return "setblock " + x + " " + y + " " + z + " " + blockId + " replace";
    }

    private void sendCommand(MinecraftClient client, String command) {
        if (client.getNetworkHandler() == null) {
            throw new IllegalStateException("Cannot send command without network handler");
        }
        client.getNetworkHandler().sendChatCommand(command);
    }

    private JsonObject toolModifyOverlay(JsonObject params) {
        String operation = requiredString(params, "op");
        JsonObject result = new JsonObject();
        result.addProperty("op", operation);

        switch (operation) {
            case "rotate" -> {
                int quarterTurns = params.has("quarters") ? requiredInt(params, "quarters") : 1;
                overlayState.setRotationQuarterTurns(overlayState.rotationQuarterTurns() + quarterTurns);
                result.addProperty("quarters", quarterTurns);
            }
            case "shift" -> {
                int dy = requiredInt(params, "dy");
                overlayState.shiftVertical(dy);
                result.addProperty("dy", dy);
            }
            case "set_anchor" -> {
                int x = requiredInt(params, "x");
                int y = requiredInt(params, "y");
                int z = requiredInt(params, "z");
                overlayState.setAnchor(new BlockPos(x, y, z));
            }
            case "replace_block" -> {
                String fromBlockId = requiredString(params, "from");
                String toBlockId = requiredString(params, "to");
                int replaced = overlayState.replaceRemainingBlock(fromBlockId, toBlockId);
                result.addProperty("from", fromBlockId);
                result.addProperty("to", toBlockId);
                result.addProperty("replaced_count", replaced);
            }
            default -> throw new IllegalArgumentException("Unsupported modify_overlay operation: " + operation);
        }

        result.add("overlay", activeOverlaySummary(overlayState.snapshot()));
        return result;
    }

    private JsonObject toolGetBlueprints(MinecraftClient client) {
        BlueprintStore store = blueprintStore(client);
        List<String> names;
        try {
            names = store.list();
        } catch (IOException error) {
            throw new RuntimeException("Failed to list blueprints", error);
        }

        JsonArray blueprintNames = new JsonArray();
        for (String name : names) {
            blueprintNames.add(name);
        }

        JsonObject result = new JsonObject();
        result.add("names", blueprintNames);
        result.addProperty("count", names.size());
        return result;
    }

    private JsonObject toolSaveBlueprint(MinecraftClient client, JsonObject params) {
        if (!overlayState.hasPlan()) {
            throw new IllegalStateException("No active overlay to save");
        }

        String name = requiredString(params, "name");
        BlueprintStore store = blueprintStore(client);
        try {
            store.save(name, overlayState.blueprintState());
        } catch (IOException error) {
            throw new RuntimeException("Failed to save blueprint '" + name + "'", error);
        }

        JsonObject result = new JsonObject();
        result.addProperty("name", name);
        result.addProperty("saved", true);
        return result;
    }

    private JsonObject toolLoadBlueprint(MinecraftClient client, JsonObject params) {
        String name = requiredString(params, "name");
        BlueprintStore store = blueprintStore(client);
        OverlayState.BlueprintState blueprintState;
        try {
            blueprintState = store.load(name);
        } catch (IOException error) {
            throw new RuntimeException("Failed to load blueprint '" + name + "'", error);
        }
        overlayState.loadBlueprint(blueprintState);

        JsonObject result = new JsonObject();
        result.addProperty("name", name);
        result.add("overlay", activeOverlaySummary(overlayState.snapshot()));
        return result;
    }

    private JsonObject blockPosToJson(BlockPos pos) {
        JsonObject json = new JsonObject();
        json.addProperty("x", pos.getX());
        json.addProperty("y", pos.getY());
        json.addProperty("z", pos.getZ());
        return json;
    }

    private int findGroundY(ClientWorld world, BlockPos playerPos) {
        int bottomY = world.getBottomY();
        for (int y = playerPos.getY() - 1; y >= bottomY; y--) {
            BlockPos scanPos = new BlockPos(playerPos.getX(), y, playerPos.getZ());
            BlockState state = world.getBlockState(scanPos);
            if (!state.isAir() && !state.getCollisionShape(world, scanPos).isEmpty()) {
                return y;
            }
        }
        return playerPos.getY() - 1;
    }

    private BlockPos requiredBlockPos(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonObject()) {
            throw new IllegalArgumentException("Expected object field: " + key);
        }

        JsonObject value = element.getAsJsonObject();
        return new BlockPos(
                requiredInt(value, "x"),
                requiredInt(value, "y"),
                requiredInt(value, "z")
        );
    }

    private JsonObject activeOverlaySummary(OverlaySnapshot snapshot) {
        JsonObject summary = new JsonObject();
        summary.addProperty("has_plan", snapshot.hasPlan());
        summary.addProperty("block_count", snapshot.transformedPlacements().size());
        summary.add("anchor", blockPosToJson(snapshot.anchor()));
        summary.addProperty("rotation_quarter_turns", snapshot.rotationQuarterTurns());
        summary.addProperty("preview_mode", snapshot.previewMode());
        summary.addProperty("confirmed", snapshot.confirmed());
        summary.addProperty("remaining_count", snapshot.remainingCount());
        return summary;
    }

    private String requiredString(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonPrimitive() || !element.getAsJsonPrimitive().isString()) {
            throw new IllegalArgumentException("Expected string field: " + key);
        }
        return element.getAsString();
    }

    private int requiredInt(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonPrimitive() || !element.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("Expected integer field: " + key);
        }
        return element.getAsInt();
    }

    private boolean requiredBoolean(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonPrimitive() || !element.getAsJsonPrimitive().isBoolean()) {
            throw new IllegalArgumentException("Expected boolean field: " + key);
        }
        return element.getAsBoolean();
    }

    private String normalizeBlockId(String blockId) {
        int stateStart = blockId.indexOf('[');
        if (stateStart < 0) {
            return blockId;
        }
        return blockId.substring(0, stateStart);
    }

    private boolean isTerrainBlock(String blockId, int y) {
        if ("minecraft:stone".equals(blockId) && y <= 63) {
            return true;
        }
        return TERRAIN_BLOCK_IDS.contains(blockId);
    }

    public static void onBuildTestCommand() {
        if (instance == null) {
            return;
        }
        instance.runBuildTestCommand(MinecraftClient.getInstance());
    }

    public static void onHudTestCommand() {
        if (instance == null) {
            return;
        }
        instance.runHudTestCommand(MinecraftClient.getInstance());
    }

    public static void onChatBuildTestCommand(String message) {
        if (instance == null) {
            return;
        }
        instance.runChatBuildTestCommand(MinecraftClient.getInstance(), message, true);
    }

    public static void onChatBuildTestCommand(String message, boolean resetOverlay) {
        if (instance == null) {
            return;
        }
        instance.runChatBuildTestCommand(MinecraftClient.getInstance(), message, resetOverlay);
    }

    private void beginBuildTestTrace(String trigger, String prompt) {
        buildTestTrigger = trigger;
        buildTestPrompt = prompt;
        resetBuildTestToolTrace();
        resetBuildTestStatusTimeline();
        buildTestStepSequence++;
        buildTestStepId = trigger + "-" + buildTestStepSequence + "-" + System.currentTimeMillis();
        buildTestStepStartedAtMillis = System.currentTimeMillis();
        buildTestStepFirstPlacementAtMillis = -1;
        buildTestStepReadyForCaptureAtMillis = -1;
        buildTestStepTickCount = 0;
    }

    private void resetBuildTestToolTrace() {
        synchronized (buildTestToolCalls) {
            buildTestToolCalls.clear();
        }
        toolCallSequence = 0;
    }

    private void resetBuildTestStatusTimeline() {
        synchronized (buildTestStatusTimeline) {
            buildTestStatusTimeline.clear();
        }
    }

    private void recordBuildTestStatusEvent(String message) {
        if (buildTestStepId == null) {
            return;
        }
        JsonObject entry = new JsonObject();
        long nowMillis = System.currentTimeMillis();
        entry.addProperty("at_ms", nowMillis);
        entry.addProperty("at", isoTimestamp(nowMillis));
        entry.addProperty("message", message);
        synchronized (buildTestStatusTimeline) {
            buildTestStatusTimeline.add(entry);
        }
    }

    private JsonArray buildTestStatusTimelineJson() {
        JsonArray timeline = new JsonArray();
        synchronized (buildTestStatusTimeline) {
            for (JsonObject event : buildTestStatusTimeline) {
                timeline.add(event.deepCopy());
            }
        }
        return timeline;
    }

    private JsonObject buildTestStepMetadata(long capturedAtMillis) {
        JsonObject step = new JsonObject();
        if (buildTestStepId != null) {
            step.addProperty("id", buildTestStepId);
        }
        step.addProperty("sequence", buildTestStepSequence);
        step.addProperty("started_at_ms", buildTestStepStartedAtMillis);
        step.addProperty("started_at", isoTimestamp(buildTestStepStartedAtMillis));
        if (buildTestStepFirstPlacementAtMillis >= 0) {
            step.addProperty("first_placement_at_ms", buildTestStepFirstPlacementAtMillis);
            step.addProperty("first_placement_at", isoTimestamp(buildTestStepFirstPlacementAtMillis));
            step.addProperty("time_to_first_placement_ms", buildTestStepFirstPlacementAtMillis - buildTestStepStartedAtMillis);
        }
        if (buildTestStepReadyForCaptureAtMillis >= 0) {
            step.addProperty("ready_for_capture_at_ms", buildTestStepReadyForCaptureAtMillis);
            step.addProperty("ready_for_capture_at", isoTimestamp(buildTestStepReadyForCaptureAtMillis));
            step.addProperty("time_to_ready_ms", buildTestStepReadyForCaptureAtMillis - buildTestStepStartedAtMillis);
        }
        step.addProperty("captured_at_ms", capturedAtMillis);
        step.addProperty("captured_at", isoTimestamp(capturedAtMillis));
        step.addProperty("elapsed_ms", capturedAtMillis - buildTestStepStartedAtMillis);
        step.addProperty("tick_count", buildTestStepTickCount);
        return step;
    }

    private void recordBuildTestToolCall(
            String tool,
            JsonObject params,
            JsonObject result,
            Throwable error,
            long startedAtMillis,
            long completedAtMillis
    ) {
        JsonObject entry = new JsonObject();
        entry.addProperty("seq", ++toolCallSequence);
        entry.addProperty("tool", tool);
        entry.addProperty("started_at_ms", startedAtMillis);
        entry.addProperty("started_at", isoTimestamp(startedAtMillis));
        entry.addProperty("completed_at_ms", completedAtMillis);
        entry.addProperty("completed_at", isoTimestamp(completedAtMillis));
        entry.addProperty("duration_ms", completedAtMillis - startedAtMillis);
        if (params != null) {
            entry.add("params", summarizeToolParams(tool, params));
        }
        if (result != null) {
            entry.add("result", summarizeToolResult(tool, result));
        }
        if (error != null) {
            entry.addProperty("error", error.getClass().getName() + ": " + error.getMessage());
        }
        synchronized (buildTestToolCalls) {
            buildTestToolCalls.add(entry);
        }
    }

    private String isoTimestamp(long epochMillis) {
        return Instant.ofEpochMilli(epochMillis).toString();
    }

    private JsonObject summarizeToolParams(String tool, JsonObject params) {
        JsonObject summary = new JsonObject();
        switch (tool) {
            case "place_blocks" -> {
                JsonElement placements = params.get("placements");
                int count = placements != null && placements.isJsonArray() ? placements.getAsJsonArray().size() : 0;
                summary.addProperty("placements_count", count);
            }
            case "inspect_area" -> {
                if (params.has("radius")) {
                    summary.addProperty("radius", params.get("radius").getAsInt());
                }
                if (params.has("detailed")) {
                    summary.addProperty("detailed", params.get("detailed").getAsBoolean());
                }
            }
            case "fill_region" -> {
                if (params.has("from_corner")) {
                    summary.add("from_corner", params.get("from_corner").getAsJsonObject().deepCopy());
                }
                if (params.has("to_corner")) {
                    summary.add("to_corner", params.get("to_corner").getAsJsonObject().deepCopy());
                }
                if (params.has("block_id")) {
                    summary.addProperty("block_id", params.get("block_id").getAsString());
                }
            }
            default -> summary = params.deepCopy();
        }
        return summary;
    }

    private JsonObject summarizeToolResult(String tool, JsonObject result) {
        JsonObject summary = new JsonObject();
        if ("place_blocks".equals(tool) || "fill_region".equals(tool)) {
            if (result.has("placed_count")) {
                summary.addProperty("placed_count", result.get("placed_count").getAsInt());
            }
            if (result.has("anchor")) {
                summary.add("anchor", result.get("anchor").getAsJsonObject().deepCopy());
            }
            return summary;
        }
        if ("inspect_area".equals(tool)) {
            if (result.has("sampled_blocks")) {
                summary.addProperty("sampled_blocks", result.get("sampled_blocks").getAsInt());
            }
            if (result.has("radius")) {
                summary.addProperty("radius", result.get("radius").getAsInt());
            }
            return summary;
        }
        return result.deepCopy();
    }

    public static Path latestBuildTestJsonPath() {
        return latestBuildTestJsonPath;
    }

    public static Path latestBuildTestScreenshotPath() {
        return latestBuildTestScreenshotPath;
    }

    public static Path latestHudTestJsonPath() {
        return latestHudTestJsonPath;
    }

    public static String latestStatusMessage() {
        if (instance == null) {
            return "";
        }
        return instance.latestStatusMessage;
    }

    private enum ChatRole {
        USER,
        ASSISTANT
    }

    private record ChatMessage(ChatRole role, String text) {
    }

    private record RenderedLine(String plainText, OrderedText orderedText, int color) {
    }

    private record HudBounds(int left, int top, int right, int bottom) {
    }

    private record HudRenderSnapshot(
            HudChatState.Mode mode,
            boolean visible,
            int screenWidth,
            int screenHeight,
            String header,
            String activeToolStatus,
            boolean assistantStreaming,
            HudBounds panelBounds,
            HudBounds inputBounds,
            List<RenderedLine> wrappedLines,
            int visibleStartIndex,
            String inputText,
            int cursorIndex,
            int cursorX
    ) {
    }

    private record HudCaptureTarget(HudChatState.Mode mode, String label) {
    }

    private record HudDebugState(
            List<ChatMessage> chatHistory,
            String activeToolStatus,
            boolean assistantStreaming,
            HudChatState.Snapshot hudState
    ) {
    }

    private static final class HudCaptureSession {
        private final long sessionTimestamp;
        private final HudDebugState previousState;
        private final List<HudCaptureTarget> targets;
        private final List<JsonObject> captures = new ArrayList<>();
        private int captureIndex;
        private int settleTicksRemaining;

        private HudCaptureSession(long sessionTimestamp, HudDebugState previousState, List<HudCaptureTarget> targets) {
            this.sessionTimestamp = sessionTimestamp;
            this.previousState = previousState;
            this.targets = targets;
        }

        private HudCaptureTarget currentTarget() {
            return targets.get(captureIndex);
        }
    }

    private record AbsolutePlacement(int x, int y, int z, String blockId) {
    }

}
