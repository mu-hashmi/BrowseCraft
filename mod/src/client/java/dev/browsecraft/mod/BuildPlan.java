package dev.browsecraft.mod;

import java.util.List;

public record BuildPlan(int totalBlocks, List<BuildPlacement> placements) {}
