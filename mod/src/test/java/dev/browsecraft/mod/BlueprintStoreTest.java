package dev.browsecraft.mod;

import net.minecraft.util.math.BlockPos;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BlueprintStoreTest {
    @TempDir
    Path tempDir;

    @Test
    void savesAndLoadsBlueprintState() throws Exception {
        Path worldSaveDir = tempDir.resolve("world");
        BlueprintStore store = new BlueprintStore(BlueprintStore.defaultDirectory(worldSaveDir));
        BuildPlan plan = new BuildPlan(
                2,
                List.of(
                        new BuildPlacement(0, 0, 0, "minecraft:stone", Map.of()),
                        new BuildPlacement(1, 1, 0, "minecraft:oak_planks", Map.of("axis", "y"))
                )
        );
        OverlayState.BlueprintState input = new OverlayState.BlueprintState(plan, new BlockPos(5, 70, 8), 3);

        Path savedPath = store.save("starter", input);
        OverlayState.BlueprintState output = store.load("starter");

        assertEquals(worldSaveDir.resolve("browsecraft").resolve("blueprints").resolve("starter.json"), savedPath);
        assertEquals(input.anchor(), output.anchor());
        assertEquals(3, output.rotationQuarterTurns());
        assertEquals(2, output.plan().totalBlocks());
        assertEquals(2, output.plan().placements().size());
        assertEquals("minecraft:oak_planks", output.plan().placements().get(1).blockId());
        assertEquals("y", output.plan().placements().get(1).blockState().get("axis"));
    }

    @Test
    void listsSavedBlueprintNamesSorted() throws Exception {
        Path worldSaveDir = tempDir.resolve("world");
        BlueprintStore store = new BlueprintStore(BlueprintStore.defaultDirectory(worldSaveDir));
        BuildPlan plan = new BuildPlan(1, List.of(new BuildPlacement(0, 0, 0, "minecraft:stone", Map.of())));
        OverlayState.BlueprintState blueprintState = new OverlayState.BlueprintState(plan, BlockPos.ORIGIN, 0);

        store.save("tower", blueprintState);
        store.save("alpha", blueprintState);

        assertEquals(List.of("alpha", "tower"), store.list());
    }
}
