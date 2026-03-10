package dev.browsecraft.mod;

import java.util.Map;

public record BuildPlacement(int dx, int dy, int dz, String blockId, Map<String, String> blockState) {}
