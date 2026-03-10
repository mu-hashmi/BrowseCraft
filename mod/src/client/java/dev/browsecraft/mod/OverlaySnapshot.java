package dev.browsecraft.mod;

import net.minecraft.util.math.BlockPos;

import java.util.List;

public record OverlaySnapshot(
        boolean hasPlan,
        boolean previewMode,
        boolean confirmed,
        BlockPos anchor,
        int rotationQuarterTurns,
        int remainingCount,
        List<OverlayState.TransformedPlacement> transformedPlacements
) {}
