package dev.browsecraft.mod;

import net.minecraft.util.math.BlockPos;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MaterialSummaryTest {

    @Test
    void aggregatesByBlockTypeSortedByCountDesc() {
        List<OverlayState.TransformedPlacement> placements = List.of(
                new OverlayState.TransformedPlacement(new BlockPos(0, 64, 0), "minecraft:stone"),
                new OverlayState.TransformedPlacement(new BlockPos(1, 64, 0), "minecraft:stone"),
                new OverlayState.TransformedPlacement(new BlockPos(2, 64, 0), "minecraft:oak_planks"),
                new OverlayState.TransformedPlacement(new BlockPos(3, 64, 0), "minecraft:stone"),
                new OverlayState.TransformedPlacement(new BlockPos(4, 64, 0), "minecraft:oak_planks")
        );

        List<MaterialSummary.MaterialCount> aggregated = MaterialSummary.aggregate(placements);

        assertEquals(2, aggregated.size());
        assertEquals("minecraft:stone", aggregated.get(0).blockId());
        assertEquals(3, aggregated.get(0).count());
        assertEquals("minecraft:oak_planks", aggregated.get(1).blockId());
        assertEquals(2, aggregated.get(1).count());
    }

    @Test
    void comparesNeededAgainstInventory() {
        List<MaterialSummary.MaterialCount> needed = List.of(
                new MaterialSummary.MaterialCount("minecraft:stone", 10),
                new MaterialSummary.MaterialCount("minecraft:oak_planks", 4)
        );

        List<MaterialSummary.MaterialDelta> deltas = MaterialSummary.compareWithInventory(
                needed,
                Map.of(
                        "minecraft:stone", 6,
                        "minecraft:oak_planks", 8
                )
        );

        assertEquals(2, deltas.size());
        assertEquals("minecraft:stone", deltas.get(0).blockId());
        assertEquals(10, deltas.get(0).needed());
        assertEquals(6, deltas.get(0).inInventory());
        assertEquals(4, deltas.get(0).missing());
        assertEquals("minecraft:oak_planks", deltas.get(1).blockId());
        assertEquals(0, deltas.get(1).missing());
    }
}
