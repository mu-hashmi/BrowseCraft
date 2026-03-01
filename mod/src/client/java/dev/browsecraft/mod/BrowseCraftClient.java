package dev.browsecraft.mod;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.brigadier.arguments.StringArgumentType;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.block.BlockState;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.gui.screen.ChatScreen;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.item.BlockItem;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.text.Text;
import net.minecraft.util.Identifier;
import net.minecraft.util.WorldSavePath;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Direction;
import org.lwjgl.glfw.GLFW;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
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
    private static volatile BrowseCraftClient instance;

    public static volatile Path latestBuildTestJsonPath;
    public static volatile Path latestBuildTestScreenshotPath;

    private OverlayState overlayState;
    private GhostRenderer ghostRenderer;
    private BuildCommandController commandController;
    private BuildBackend backend;
    private ExecutorService workerExecutor;
    private String clientId;
    private final WorldIdResolver worldIdResolver = new WorldIdResolver();

    private KeyBinding openBuildKey;
    private KeyBinding rotateKey;
    private KeyBinding shiftUpKey;
    private KeyBinding shiftDownKey;
    private KeyBinding confirmKey;
    private KeyBinding cancelKey;
    private KeyBinding.Category keyCategory;

    private int buildTestCaptureTicksRemaining = -1;
    private int inventoryPollCountdown;

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
                () -> worldIdResolver.resolve(MinecraftClient.getInstance())
        );

        registerKeyBindings();
        registerCommands();
        registerClientTick();
        ghostRenderer.register();
    }

    private void registerKeyBindings() {
        this.keyCategory = KeyBinding.Category.create(Identifier.of("browsecraft", "controls"));
        this.openBuildKey = KeyBindingHelper.registerKeyBinding(new KeyBinding(
                "key.browsecraft.open_build",
                InputUtil.Type.KEYSYM,
                GLFW.GLFW_KEY_G,
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
            dispatcher.register(literal("build")
                    .then(argument("query", StringArgumentType.greedyString())
                            .executes(context -> {
                                String query = StringArgumentType.getString(context, "query");
                                commandController.submit(query);
                                return 1;
                            })));

            dispatcher.register(literal("build-test")
                    .executes(context -> {
                        runBuildTestCommand(MinecraftClient.getInstance());
                        return 1;
                    }));

            dispatcher.register(literal("imagine")
                    .then(literal("modify")
                            .then(argument("prompt", StringArgumentType.greedyString())
                                    .executes(context -> {
                                        String prompt = StringArgumentType.getString(context, "prompt");
                                        commandController.submitImagineModify(prompt);
                                        return 1;
                                    })))
                    .then(argument("prompt", StringArgumentType.greedyString())
                            .executes(context -> {
                                String prompt = StringArgumentType.getString(context, "prompt");
                                commandController.submitImagine(prompt);
                                return 1;
                            })));

            dispatcher.register(literal("chat")
                    .then(argument("message", StringArgumentType.greedyString())
                            .executes(context -> {
                                String message = StringArgumentType.getString(context, "message");
                                commandController.submitChat(message);
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
            if (openBuildKey.wasPressed()) {
                client.setScreen(new ChatScreen("/build ", false));
            }

            if (client.player == null || client.world == null) {
                ghostRenderer.setAvailableBlockTypes(Set.of());
                return;
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

            if (buildTestCaptureTicksRemaining >= 0) {
                buildTestCaptureTicksRemaining--;
                if (buildTestCaptureTicksRemaining == 0) {
                    captureBuildTestArtifacts(client);
                }
            }
        });
    }

    private void runBuildTestCommand(MinecraftClient client) {
        if (client.player == null) {
            return;
        }

        BuildPlan testPlan = createBuildTestPlan();
        overlayState.setPlan(testPlan);
        overlayState.setRotationQuarterTurns(rotationForFacing(client.player.getHorizontalFacing()));
        buildTestCaptureTicksRemaining = 5;
        client.player.sendMessage(Text.literal("Loaded /build-test plan"), true);
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

    private void captureBuildTestArtifacts(MinecraftClient client) {
        try {
            OverlaySnapshot snapshot = overlayState.snapshot();
            GhostRenderer.RenderFrameSnapshot renderSnapshot = ghostRenderer.latestFrameSnapshot();

            Path runDir = client.runDirectory.toPath();
            Path buildTestDir = runDir.resolve("browsecraft").resolve("build-test");
            Path screenshotsDir = runDir.resolve("screenshots");
            Files.createDirectories(buildTestDir);
            Files.createDirectories(screenshotsDir);

            Path jsonPath = buildTestDir.resolve("ghost-state.json");
            Path screenshotPath = screenshotsDir.resolve("browsecraft-build-test-" + Instant.now().toEpochMilli() + ".png");

            JsonObject root = new JsonObject();
            root.add("overlay", overlayToJson(snapshot));
            root.add("validation", validationToJson(snapshot));

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

            Files.writeString(jsonPath, GSON.toJson(root));
            Files.write(screenshotPath, new byte[0]);

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
        return switch (tool) {
            case "player_position" -> toolPlayerPosition(client);
            case "player_inventory" -> toolPlayerInventory(client);
            case "inspect_area" -> toolInspectArea(client, params);
            case "get_active_overlay" -> activeOverlaySummary(overlayState.snapshot());
            case "modify_overlay" -> toolModifyOverlay(params);
            case "get_blueprints" -> toolGetBlueprints(client);
            case "save_blueprint" -> toolSaveBlueprint(client, params);
            case "load_blueprint" -> toolLoadBlueprint(client, params);
            default -> throw new IllegalArgumentException("Unsupported tool: " + tool);
        };
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
        int radius = Math.min(16, Math.max(0, requestedRadius));
        ClientWorld world = client.world;
        Map<String, Integer> blockCounts = new TreeMap<>();
        int sampledBlocks = 0;

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    BlockPos pos = center.add(dx, dy, dz);
                    String blockId = Registries.BLOCK.getId(world.getBlockState(pos).getBlock()).toString();
                    blockCounts.merge(blockId, 1, Integer::sum);
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
        return result;
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

    public static void onBuildTestCommand() {
        if (instance == null) {
            return;
        }
        instance.runBuildTestCommand(MinecraftClient.getInstance());
    }

    public static Path latestBuildTestJsonPath() {
        return latestBuildTestJsonPath;
    }

    public static Path latestBuildTestScreenshotPath() {
        return latestBuildTestScreenshotPath;
    }
}
