package dev.browsecraft.mod;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.util.math.BlockPos;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public final class BlueprintStore {
    private static final Pattern VALID_NAME = Pattern.compile("[A-Za-z0-9._-]+");
    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private final Path directory;

    public BlueprintStore(Path directory) {
        this.directory = directory;
    }

    public static Path defaultDirectory(Path worldSaveDir) {
        return worldSaveDir.resolve("browsecraft").resolve("blueprints");
    }

    public Path save(String name, OverlayState.BlueprintState blueprintState) throws IOException {
        Path path = pathFor(name);
        Files.createDirectories(directory);
        Files.writeString(path, GSON.toJson(toJson(blueprintState)));
        return path;
    }

    public OverlayState.BlueprintState load(String name) throws IOException {
        Path path = pathFor(name);
        String json = Files.readString(path);
        return fromJson(JsonParser.parseString(json).getAsJsonObject());
    }

    public List<String> list() throws IOException {
        if (!Files.exists(directory)) {
            return List.of();
        }

        try (Stream<Path> files = Files.list(directory)) {
            return files
                    .filter(Files::isRegularFile)
                    .map(Path::getFileName)
                    .map(Path::toString)
                    .filter(name -> name.endsWith(".json"))
                    .map(name -> name.substring(0, name.length() - 5))
                    .sorted(Comparator.naturalOrder())
                    .toList();
        }
    }

    private Path pathFor(String name) {
        if (!VALID_NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("Blueprint name must match [A-Za-z0-9._-]+");
        }
        return directory.resolve(name + ".json");
    }

    private JsonObject toJson(OverlayState.BlueprintState blueprintState) {
        JsonObject root = new JsonObject();
        root.addProperty("rotation_quarter_turns", blueprintState.rotationQuarterTurns());
        root.add("anchor", blockPosToJson(blueprintState.anchor()));
        root.add("plan", planToJson(blueprintState.plan()));
        return root;
    }

    private JsonObject planToJson(BuildPlan plan) {
        JsonObject json = new JsonObject();
        json.addProperty("total_blocks", plan.totalBlocks());
        var placements = new com.google.gson.JsonArray();
        for (BuildPlacement placement : plan.placements()) {
            JsonObject placementJson = new JsonObject();
            placementJson.addProperty("dx", placement.dx());
            placementJson.addProperty("dy", placement.dy());
            placementJson.addProperty("dz", placement.dz());
            placementJson.addProperty("block_id", placement.blockId());

            JsonObject blockState = new JsonObject();
            for (Map.Entry<String, String> entry : placement.blockState().entrySet()) {
                blockState.addProperty(entry.getKey(), entry.getValue());
            }
            placementJson.add("block_state", blockState);
            placements.add(placementJson);
        }
        json.add("placements", placements);
        return json;
    }

    private OverlayState.BlueprintState fromJson(JsonObject root) {
        BuildPlan plan = parsePlan(requiredObject(root, "plan"));
        BlockPos anchor = parseBlockPos(requiredObject(root, "anchor"));
        int rotationQuarterTurns = requiredInt(root, "rotation_quarter_turns");
        return new OverlayState.BlueprintState(plan, anchor, rotationQuarterTurns);
    }

    private BuildPlan parsePlan(JsonObject json) {
        int totalBlocks = requiredInt(json, "total_blocks");
        List<BuildPlacement> placements = new ArrayList<>();
        for (JsonElement element : requiredArray(json, "placements")) {
            JsonObject placement = element.getAsJsonObject();
            JsonObject blockStateObject = requiredObject(placement, "block_state");
            Map<String, String> blockState = new HashMap<>();
            for (Map.Entry<String, JsonElement> entry : blockStateObject.entrySet()) {
                blockState.put(entry.getKey(), entry.getValue().getAsString());
            }
            placements.add(new BuildPlacement(
                    requiredInt(placement, "dx"),
                    requiredInt(placement, "dy"),
                    requiredInt(placement, "dz"),
                    requiredString(placement, "block_id"),
                    Map.copyOf(blockState)
            ));
        }
        return new BuildPlan(totalBlocks, List.copyOf(placements));
    }

    private BlockPos parseBlockPos(JsonObject json) {
        return new BlockPos(requiredInt(json, "x"), requiredInt(json, "y"), requiredInt(json, "z"));
    }

    private JsonObject blockPosToJson(BlockPos pos) {
        JsonObject json = new JsonObject();
        json.addProperty("x", pos.getX());
        json.addProperty("y", pos.getY());
        json.addProperty("z", pos.getZ());
        return json;
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

    private com.google.gson.JsonArray requiredArray(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonArray()) {
            throw new IllegalArgumentException("Expected array field: " + key);
        }
        return element.getAsJsonArray();
    }

    private JsonObject requiredObject(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonObject()) {
            throw new IllegalArgumentException("Expected object field: " + key);
        }
        return element.getAsJsonObject();
    }
}
