package dev.browsecraft.mod;

import net.minecraft.util.math.BlockPos;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OverlayStateTest {

    @Test
    void rotatesAcrossAllQuadrants() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(
                1,
                List.of(new BuildPlacement(1, 0, 2, "minecraft:stone", Map.of()))
        );

        state.setPlan(plan);
        state.setAnchor(new BlockPos(10, 64, 20));

        assertEquals(new BlockPos(11, 64, 22), state.transformedPlacements().getFirst().pos());

        state.rotateClockwise();
        assertEquals(new BlockPos(8, 64, 21), state.transformedPlacements().getFirst().pos());

        state.rotateClockwise();
        assertEquals(new BlockPos(9, 64, 18), state.transformedPlacements().getFirst().pos());

        state.rotateClockwise();
        assertEquals(new BlockPos(12, 64, 19), state.transformedPlacements().getFirst().pos());
    }

    @Test
    void removesSatisfiedBlocksWhenConfirmed() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(
                2,
                List.of(
                        new BuildPlacement(0, 0, 0, "minecraft:stone", Map.of()),
                        new BuildPlacement(1, 0, 0, "minecraft:dirt", Map.of())
                )
        );

        state.setPlan(plan);
        state.setAnchor(BlockPos.ORIGIN);
        state.confirm();

        int removed = state.updateSatisfiedBlocks(pos -> {
            if (pos.equals(new BlockPos(0, 0, 0))) {
                return "minecraft:stone";
            }
            return "minecraft:air";
        });

        assertEquals(1, removed);
        assertEquals(1, state.remainingCount());
    }

    @Test
    void snapshotReflectsState() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(
                1,
                List.of(new BuildPlacement(0, 1, 0, "minecraft:cobblestone", Map.of()))
        );

        state.setPlan(plan);
        state.setAnchor(new BlockPos(4, 70, 4));
        OverlaySnapshot snapshot = state.snapshot();

        assertTrue(snapshot.hasPlan());
        assertTrue(snapshot.previewMode());
        assertFalse(snapshot.confirmed());
        assertEquals(new BlockPos(4, 70, 4), snapshot.anchor());
        assertEquals(0, snapshot.rotationQuarterTurns());
        assertEquals(1, snapshot.remainingCount());
        assertEquals(new BlockPos(4, 71, 4), snapshot.transformedPlacements().getFirst().pos());
    }

    @Test
    void replaceRemainingBlockUpdatesMatchingPlacements() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(
                3,
                List.of(
                        new BuildPlacement(0, 0, 0, "minecraft:stone", Map.of()),
                        new BuildPlacement(1, 0, 0, "minecraft:dirt", Map.of()),
                        new BuildPlacement(2, 0, 0, "minecraft:stone", Map.of())
                )
        );

        state.setPlan(plan);
        int replaced = state.replaceRemainingBlock("minecraft:stone", "minecraft:deepslate");

        assertEquals(2, replaced);
        List<OverlayState.TransformedPlacement> transformed = state.transformedPlacements();
        assertEquals("minecraft:deepslate", transformed.get(0).blockId());
        assertEquals("minecraft:dirt", transformed.get(1).blockId());
        assertEquals("minecraft:deepslate", transformed.get(2).blockId());
    }

    @Test
    void blueprintStateKeepsOriginalPlanAfterConfirmationProgress() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(
                2,
                List.of(
                        new BuildPlacement(0, 0, 0, "minecraft:stone", Map.of()),
                        new BuildPlacement(1, 0, 0, "minecraft:dirt", Map.of())
                )
        );

        state.setPlan(plan);
        state.setAnchor(new BlockPos(5, 70, 5));
        state.confirm();
        state.updateSatisfiedBlocks(pos -> {
            if (pos.equals(new BlockPos(5, 70, 5))) {
                return "minecraft:stone";
            }
            return "minecraft:air";
        });

        OverlayState.BlueprintState blueprintState = state.blueprintState();
        assertEquals(2, blueprintState.plan().placements().size());
        assertEquals(new BlockPos(5, 70, 5), blueprintState.anchor());
        assertEquals(0, blueprintState.rotationQuarterTurns());
    }

    @Test
    void loadBlueprintRestoresAnchorAndRotationInPreviewMode() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(
                1,
                List.of(new BuildPlacement(2, 1, 0, "minecraft:oak_planks", Map.of()))
        );

        state.loadBlueprint(new OverlayState.BlueprintState(plan, new BlockPos(10, 64, 10), 1));
        state.applyInitialPreviewAnchor(new BlockPos(1, 64, 1));

        assertTrue(state.hasPlan());
        assertTrue(state.isPreviewMode());
        assertFalse(state.isConfirmed());
        assertEquals(new BlockPos(10, 64, 10), state.anchor());
        assertEquals(1, state.rotationQuarterTurns());
        assertEquals(new BlockPos(10, 65, 12), state.transformedPlacements().getFirst().pos());
    }

    @Test
    void autoPreviewAnchorTracksPlayerUntilManuallyPinned() {
        OverlayState state = new OverlayState();
        BuildPlan plan = new BuildPlan(1, List.of(new BuildPlacement(0, 0, 0, "minecraft:stone", Map.of())));

        state.setPlan(plan);
        state.applyInitialPreviewAnchor(new BlockPos(3, 64, 3));
        state.applyInitialPreviewAnchor(new BlockPos(8, 64, 8));

        assertEquals(new BlockPos(8, 64, 8), state.anchor());

        state.setAnchor(new BlockPos(10, 64, 10));
        state.applyInitialPreviewAnchor(new BlockPos(1, 64, 1));
        assertEquals(new BlockPos(10, 64, 10), state.anchor());
    }
}
