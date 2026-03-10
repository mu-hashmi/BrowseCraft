package dev.browsecraft.mod;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

public final class BrowseCraftHudClientGameTests implements FabricClientGameTest {
    @Override
    public void runTest(ClientGameTestContext context) {
        String suite = System.getProperty("browsecraft.clientGameTestSuite");
        if (suite != null && !"hud".equals(suite)) {
            return;
        }

        TestSingleplayerContext singleplayer = context.worldBuilder().create();
        context.waitTicks(40);

        context.runOnClient(client -> BrowseCraftClient.onHudTestCommand());

        context.waitFor(client -> {
            Path jsonPath = BrowseCraftClient.latestHudTestJsonPath();
            return jsonPath != null && Files.exists(jsonPath) && hudArtifactsReady(jsonPath);
        }, 2400);

        Path jsonPath = BrowseCraftClient.latestHudTestJsonPath();
        if (jsonPath == null) {
            throw new AssertionError("hud-test manifest was not captured");
        }

        assertHudArtifacts(jsonPath);

        singleplayer.close();
        context.waitFor(client -> client.getServer() == null, 300);
    }

    private static boolean hudArtifactsReady(Path jsonPath) {
        try {
            JsonObject root = JsonParser.parseString(Files.readString(jsonPath)).getAsJsonObject();
            JsonArray captures = root.getAsJsonArray("captures");
            if (captures == null || captures.size() != 3) {
                return false;
            }
            for (int index = 0; index < captures.size(); index++) {
                JsonObject capture = captures.get(index).getAsJsonObject();
                Path screenshotPath = Path.of(capture.get("screenshot_path").getAsString());
                if (!Files.exists(screenshotPath) || Files.size(screenshotPath) == 0) {
                    return false;
                }
            }
            return true;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static void assertHudArtifacts(Path jsonPath) {
        try {
            JsonObject root = JsonParser.parseString(Files.readString(jsonPath)).getAsJsonObject();
            JsonArray captures = root.getAsJsonArray("captures");
            if (captures == null || captures.size() != 3) {
                throw new AssertionError("expected exactly three hud captures");
            }

            Set<String> modes = new HashSet<>();
            for (int index = 0; index < captures.size(); index++) {
                JsonObject capture = captures.get(index).getAsJsonObject();
                String mode = capture.get("mode").getAsString();
                modes.add(mode);

                Path screenshotPath = Path.of(capture.get("screenshot_path").getAsString());
                if (!Files.exists(screenshotPath) || Files.size(screenshotPath) == 0) {
                    throw new AssertionError("missing screenshot for mode " + mode);
                }

                boolean visible = capture.get("visible").getAsBoolean();
                JsonArray wrappedLines = capture.getAsJsonArray("wrapped_lines");
                if ("HIDDEN".equals(mode)) {
                    if (visible) {
                        throw new AssertionError("hidden hud capture should not be visible");
                    }
                } else {
                    if (!visible) {
                        throw new AssertionError("visible hud capture unexpectedly hidden for mode " + mode);
                    }
                    if (capture.get("panel_bounds") == null || wrappedLines == null || wrappedLines.size() == 0) {
                        throw new AssertionError("missing panel metadata for mode " + mode);
                    }
                }

                if ("INPUT".equals(mode) && capture.get("input_bounds") == null) {
                    throw new AssertionError("input capture missing input bar bounds");
                }
            }

            if (!modes.equals(Set.of("HIDDEN", "HUD", "INPUT"))) {
                throw new AssertionError("unexpected hud capture modes: " + modes);
            }
        } catch (Exception error) {
            throw new RuntimeException("failed to validate hud-test manifest", error);
        }
    }
}
