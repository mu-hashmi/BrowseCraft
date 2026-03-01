package dev.browsecraft.mod;

import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;

import java.nio.file.Files;
import java.nio.file.Path;

public final class BrowseCraftClientGameTests implements FabricClientGameTest {
    @Override
    public void runTest(ClientGameTestContext context) {
        TestSingleplayerContext singleplayer = context.worldBuilder().create();
        context.waitTicks(40);

        context.runOnClient(client -> BrowseCraftClient.onBuildTestCommand());

        context.waitFor(client -> {
            String latestStatus = BrowseCraftClient.latestStatusMessage();
            if (latestStatus.startsWith("build submit failed: ")) {
                throw new AssertionError(latestStatus);
            }
            if (latestStatus.startsWith("CONFIG_ERROR: ")) {
                throw new AssertionError(latestStatus);
            }
            if (latestStatus.startsWith("INTERNAL_ERROR: ")) {
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

        try {
            String json = Files.readString(jsonPath);
            if (!json.contains("\"passed\": true")) {
                throw new AssertionError("validation.passed was not true in artifact JSON");
            }
            if (Files.size(screenshotPath) == 0) {
                throw new AssertionError("build-test screenshot is empty");
            }
        } catch (Exception error) {
            throw new RuntimeException("Failed to read build-test artifact", error);
        }

        singleplayer.close();
        context.waitFor(client -> client.getServer() == null, 300);
    }

    private static boolean hasImageData(Path path) {
        try {
            return Files.exists(path) && Files.size(path) > 0;
        } catch (Exception ignored) {
            return false;
        }
    }
}
