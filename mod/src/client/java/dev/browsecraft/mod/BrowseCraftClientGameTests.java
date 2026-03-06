package dev.browsecraft.mod;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public final class BrowseCraftClientGameTests implements FabricClientGameTest {
    private static final String STEP1_PROMPT =
            "Build a 5x5 minecraft:stone room centered on your current block position. "
                    + "Make 3-block-high walls, keep the interior hollow, and do not add a roof.";
    private static final String STEP2_PROMPT =
            "Modify that room by adding a 2-block-tall minecraft:oak_door on the south wall at ground level. "
                    + "Keep every other wall block unchanged and use block placement tools.";
    private static final String STEP3_PROMPT =
            "Replace all minecraft:stone blocks in that room with minecraft:birch_planks, "
                    + "keeping the same shape and keeping the door. "
                    + "Do not just describe it; call block placement tools to apply the replacement.";

    @Override
    public void runTest(ClientGameTestContext context) {
        String suite = System.getProperty("browsecraft.clientGameTestSuite");
        if ("hud".equals(suite)) {
            return;
        }

        TestSingleplayerContext singleplayer = context.worldBuilder().create();
        context.waitTicks(40);

        StepArtifact step1 = runChatStep(context, STEP1_PROMPT, true);
        StructureSnapshot room = readStructureSnapshot(step1.jsonPath());
        assertRoomLikeShape(room, "minecraft:stone");

        StepArtifact step2 = runChatStep(context, STEP2_PROMPT, false);
        StructureSnapshot withDoor = readStructureSnapshot(step2.jsonPath());
        assertDoorOnFrontWall(withDoor);
        int doorCount = countDoorBlocks(withDoor.blocksByPos());
        if (doorCount < 2) {
            throw new AssertionError("expected a 2-block door after step 2");
        }

        StepArtifact step3 = runChatStep(context, STEP3_PROMPT, false);
        StructureSnapshot replaced = readStructureSnapshot(step3.jsonPath());
        assertRoomLikeShape(replaced, "minecraft:birch_planks");
        if (countMaterial(replaced.blocksByPos(), "minecraft:stone") != 0) {
            throw new AssertionError("stone blocks remained after replacement step");
        }

        Set<String> step2Positions = new HashSet<>(withDoor.blocksByPos().keySet());
        Set<String> step3Positions = new HashSet<>(replaced.blocksByPos().keySet());
        if (!step2Positions.equals(step3Positions)) {
            throw new AssertionError("shape changed between step 2 and step 3");
        }

        singleplayer.close();
        context.waitFor(client -> client.getServer() == null, 300);
    }

    private static StepArtifact runChatStep(ClientGameTestContext context, String prompt, boolean resetOverlay) {
        context.runOnClient(client -> BrowseCraftClient.onChatBuildTestCommand(prompt, resetOverlay));

        context.waitFor(client -> {
            String latestStatus = BrowseCraftClient.latestStatusMessage();
            if (latestStatus.startsWith("build submit failed: ")) {
                throw new AssertionError(latestStatus);
            }
            if (latestStatus.startsWith("chat submit failed: ")) {
                throw new AssertionError(latestStatus);
            }
            if (latestStatus.startsWith("CONFIG_ERROR: ")) {
                throw new AssertionError(latestStatus);
            }
            if (latestStatus.startsWith("INTERNAL_ERROR: ")) {
                throw new AssertionError(latestStatus);
            }
            if (latestStatus.startsWith("CHAT_BUILD_TEST_TIMEOUT: ")) {
                throw new AssertionError(latestStatus);
            }

            Path jsonPath = BrowseCraftClient.latestBuildTestJsonPath();
            Path screenshotPath = BrowseCraftClient.latestBuildTestScreenshotPath();
            return jsonPath != null
                    && screenshotPath != null
                    && Files.exists(jsonPath)
                    && hasImageData(screenshotPath);
        }, 6000);

        Path jsonPath = BrowseCraftClient.latestBuildTestJsonPath();
        Path screenshotPath = BrowseCraftClient.latestBuildTestScreenshotPath();
        if (jsonPath == null || screenshotPath == null) {
            throw new AssertionError("build-test artifacts were not captured");
        }

        context.waitFor(client -> {
            String latestStatus = BrowseCraftClient.latestStatusMessage();
            if (latestStatus.startsWith("build submit failed: ")
                    || latestStatus.startsWith("chat submit failed: ")
                    || latestStatus.startsWith("CONFIG_ERROR: ")
                    || latestStatus.startsWith("INTERNAL_ERROR: ")) {
                throw new AssertionError(latestStatus);
            }
            return latestStatus.startsWith("chat: ");
        }, 3000);
        context.waitTicks(20);

        return new StepArtifact(jsonPath, screenshotPath);
    }

    private static StructureSnapshot readStructureSnapshot(Path jsonPath) {
        try {
            JsonObject root = JsonParser.parseString(Files.readString(jsonPath)).getAsJsonObject();
            JsonObject overlay = root.getAsJsonObject("overlay");
            JsonObject player = root.getAsJsonObject("player");
            JsonArray placements = overlay.getAsJsonArray("transformed_placements");

            Map<String, String> blocksByPos = new HashMap<>();
            int minX = Integer.MAX_VALUE;
            int maxX = Integer.MIN_VALUE;
            int minY = Integer.MAX_VALUE;
            int maxY = Integer.MIN_VALUE;
            int minZ = Integer.MAX_VALUE;
            int maxZ = Integer.MIN_VALUE;

            for (JsonElement element : placements) {
                JsonObject placement = element.getAsJsonObject();
                int x = placement.get("x").getAsInt();
                int y = placement.get("y").getAsInt();
                int z = placement.get("z").getAsInt();
                String blockId = placement.get("block_id").getAsString();
                blocksByPos.put(key(x, y, z), blockId);

                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
                minZ = Math.min(minZ, z);
                maxZ = Math.max(maxZ, z);
            }

            if (blocksByPos.isEmpty()) {
                throw new AssertionError("overlay placements were empty");
            }

            String facing = player.get("facing").getAsString();
            return new StructureSnapshot(blocksByPos, minX, maxX, minY, maxY, minZ, maxZ, facing);
        } catch (Exception error) {
            throw new RuntimeException("failed to parse build-test artifact", error);
        }
    }

    private static void assertRoomLikeShape(StructureSnapshot snapshot, String expectedMaterial) {
        int width = (snapshot.maxX() - snapshot.minX()) + 1;
        int depth = (snapshot.maxZ() - snapshot.minZ()) + 1;
        int height = (snapshot.maxY() - snapshot.minY()) + 1;
        if (width < 5 || depth < 5 || height < 3) {
            throw new AssertionError("expected at least a 5x5x3 room footprint");
        }

        int expectedMaterialCount = countMaterial(snapshot.blocksByPos(), expectedMaterial);
        if (expectedMaterialCount < 36) {
            throw new AssertionError("not enough " + expectedMaterial + " blocks for room walls");
        }
    }

    private static void assertDoorOnFrontWall(StructureSnapshot snapshot) {
        Set<String> doorPositions = new HashSet<>();
        for (Map.Entry<String, String> entry : snapshot.blocksByPos().entrySet()) {
            if (entry.getValue().endsWith("_door")) {
                doorPositions.add(entry.getKey());
            }
        }
        if (doorPositions.isEmpty()) {
            throw new AssertionError("expected a door block on step 2");
        }

        int wallCoordinate;
        boolean alongZ;
        switch (snapshot.facing()) {
            case "north" -> {
                wallCoordinate = snapshot.minZ();
                alongZ = true;
            }
            case "south" -> {
                wallCoordinate = snapshot.maxZ();
                alongZ = true;
            }
            case "west" -> {
                wallCoordinate = snapshot.minX();
                alongZ = false;
            }
            case "east" -> {
                wallCoordinate = snapshot.maxX();
                alongZ = false;
            }
            default -> throw new AssertionError("unsupported facing: " + snapshot.facing());
        }

        for (String doorPos : doorPositions) {
            String[] parts = doorPos.split(",");
            int x = Integer.parseInt(parts[0]);
            int y = Integer.parseInt(parts[1]);
            int z = Integer.parseInt(parts[2]);
            boolean onFrontWall = alongZ ? (z == wallCoordinate) : (x == wallCoordinate);
            boolean nearGround = y <= snapshot.minY() + 1;
            if (onFrontWall && nearGround) {
                return;
            }
        }
        throw new AssertionError("door not placed on front-facing wall near ground");
    }

    private static int countDoorBlocks(Map<String, String> blocksByPos) {
        int count = 0;
        for (String blockId : blocksByPos.values()) {
            if (blockId.endsWith("_door")) {
                count++;
            }
        }
        return count;
    }

    private static int countMaterial(Map<String, String> blocksByPos, String blockId) {
        int count = 0;
        for (String value : blocksByPos.values()) {
            if (value.equals(blockId)) {
                count++;
            }
        }
        return count;
    }

    private static String key(int x, int y, int z) {
        return x + "," + y + "," + z;
    }

    private static boolean hasImageData(Path path) {
        try {
            return Files.exists(path) && Files.size(path) > 0;
        } catch (Exception ignored) {
            return false;
        }
    }

    private record StepArtifact(Path jsonPath, Path screenshotPath) {
    }

    private record StructureSnapshot(
            Map<String, String> blocksByPos,
            int minX,
            int maxX,
            int minY,
            int maxY,
            int minZ,
            int maxZ,
            String facing
    ) {
    }
}
