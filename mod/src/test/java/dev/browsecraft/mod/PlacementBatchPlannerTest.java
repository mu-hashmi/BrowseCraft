package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PlacementBatchPlannerTest {
    @Test
    void contiguousLineUsesSingleFillCuboid() {
        PlacementBatchPlanner.Plan plan = PlacementBatchPlanner.plan(List.of(
                new PlacementBatchPlanner.Placement(1, 64, 1, "minecraft:stone"),
                new PlacementBatchPlanner.Placement(2, 64, 1, "minecraft:stone")
        ));

        assertEquals(1, plan.fillCuboids().size());
        PlacementBatchPlanner.Cuboid cuboid = plan.fillCuboids().getFirst();
        assertEquals("minecraft:stone", cuboid.blockId());
        assertEquals(1, cuboid.minX());
        assertEquals(2, cuboid.maxX());
        assertEquals(64, cuboid.minY());
        assertEquals(2, cuboid.volume());
        assertEquals(0, plan.setBlocks().size());
    }

    @Test
    void singletonPlacementStaysSetblock() {
        PlacementBatchPlanner.Plan plan = PlacementBatchPlanner.plan(List.of(
                new PlacementBatchPlanner.Placement(5, 70, 9, "minecraft:oak_planks")
        ));

        assertEquals(0, plan.fillCuboids().size());
        assertEquals(1, plan.setBlocks().size());
        assertEquals(5, plan.setBlocks().getFirst().x());
    }

    @Test
    void lShapeExtractsCuboidAndLeavesResidualSetblock() {
        PlacementBatchPlanner.Plan plan = PlacementBatchPlanner.plan(List.of(
                new PlacementBatchPlanner.Placement(0, 64, 0, "minecraft:stone"),
                new PlacementBatchPlanner.Placement(1, 64, 0, "minecraft:stone"),
                new PlacementBatchPlanner.Placement(0, 64, 1, "minecraft:stone")
        ));

        assertEquals(1, plan.fillCuboids().size());
        assertEquals(1, plan.setBlocks().size());
        assertEquals(0, plan.setBlocks().getFirst().x());
        assertEquals(1, plan.setBlocks().getFirst().z());
    }

    @Test
    void differentBlockTypesArePlannedSeparately() {
        PlacementBatchPlanner.Plan plan = PlacementBatchPlanner.plan(List.of(
                new PlacementBatchPlanner.Placement(0, 64, 0, "minecraft:stone"),
                new PlacementBatchPlanner.Placement(1, 64, 0, "minecraft:stone"),
                new PlacementBatchPlanner.Placement(5, 64, 5, "minecraft:oak_planks"),
                new PlacementBatchPlanner.Placement(5, 65, 5, "minecraft:oak_planks")
        ));

        assertEquals(2, plan.fillCuboids().size());
        assertEquals(0, plan.setBlocks().size());
    }
}
