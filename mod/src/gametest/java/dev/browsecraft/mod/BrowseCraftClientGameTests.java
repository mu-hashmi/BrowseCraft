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
            Path jsonPath = BrowseCraftClient.latestBuildTestJsonPath();
            Path screenshotPath = BrowseCraftClient.latestBuildTestScreenshotPath();
            return jsonPath != null
                    && screenshotPath != null
                    && Files.exists(jsonPath)
                    && Files.exists(screenshotPath);
        }, 300);

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
        } catch (Exception error) {
            throw new RuntimeException("Failed to read build-test artifact", error);
        }

        singleplayer.close();
        context.waitFor(client -> client.getServer() == null, 300);
    }
}
