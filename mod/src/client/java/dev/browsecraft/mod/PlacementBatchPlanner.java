package dev.browsecraft.mod;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

final class PlacementBatchPlanner {
    private static final Comparator<PositionKey> POSITION_ORDER =
            Comparator.comparingInt(PositionKey::y)
                    .thenComparingInt(PositionKey::x)
                    .thenComparingInt(PositionKey::z);

    private static final Comparator<Cuboid> CUBOID_ORDER =
            Comparator.comparingInt(Cuboid::minY)
                    .thenComparingInt(Cuboid::minX)
                    .thenComparingInt(Cuboid::minZ)
                    .thenComparing(Cuboid::blockId);

    private static final Comparator<Placement> PLACEMENT_ORDER =
            Comparator.comparingInt(Placement::y)
                    .thenComparingInt(Placement::x)
                    .thenComparingInt(Placement::z)
                    .thenComparing(Placement::blockId);

    private PlacementBatchPlanner() {
    }

    static Plan plan(List<Placement> placements) {
        if (placements.isEmpty()) {
            return new Plan(List.of(), List.of());
        }

        Map<PositionKey, String> blockByPos = new LinkedHashMap<>();
        for (Placement placement : placements) {
            blockByPos.put(new PositionKey(placement.x(), placement.y(), placement.z()), placement.blockId());
        }

        Map<String, Set<PositionKey>> remainingByBlock = new TreeMap<>();
        for (Map.Entry<PositionKey, String> entry : blockByPos.entrySet()) {
            remainingByBlock.computeIfAbsent(entry.getValue(), ignored -> new HashSet<>()).add(entry.getKey());
        }

        List<Cuboid> fillCuboids = new ArrayList<>();
        List<Placement> setBlocks = new ArrayList<>();

        for (Map.Entry<String, Set<PositionKey>> byBlock : remainingByBlock.entrySet()) {
            String blockId = byBlock.getKey();
            Set<PositionKey> remaining = byBlock.getValue();
            while (!remaining.isEmpty()) {
                PositionKey seed = smallestPosition(remaining);
                Cuboid cuboid = growCuboid(seed, blockId, remaining);
                removeCuboid(cuboid, remaining);
                if (cuboid.volume() >= 2) {
                    fillCuboids.add(cuboid);
                } else {
                    setBlocks.add(new Placement(seed.x(), seed.y(), seed.z(), blockId));
                }
            }
        }

        fillCuboids.sort(CUBOID_ORDER);
        setBlocks.sort(PLACEMENT_ORDER);
        return new Plan(List.copyOf(fillCuboids), List.copyOf(setBlocks));
    }

    private static PositionKey smallestPosition(Set<PositionKey> positions) {
        PositionKey smallest = null;
        for (PositionKey position : positions) {
            if (smallest == null || POSITION_ORDER.compare(position, smallest) < 0) {
                smallest = position;
            }
        }
        if (smallest == null) {
            throw new IllegalStateException("positions must not be empty");
        }
        return smallest;
    }

    private static Cuboid growCuboid(PositionKey seed, String blockId, Set<PositionKey> positions) {
        int minX = seed.x();
        int maxX = seed.x();
        int minY = seed.y();
        int maxY = seed.y();
        int minZ = seed.z();
        int maxZ = seed.z();

        while (positions.contains(new PositionKey(maxX + 1, minY, minZ))) {
            maxX++;
        }
        while (hasFullZSlice(positions, minY, minX, maxX, maxZ + 1)) {
            maxZ++;
        }
        while (hasFullYSlice(positions, maxY + 1, minX, maxX, minZ, maxZ)) {
            maxY++;
        }

        return new Cuboid(blockId, minX, minY, minZ, maxX, maxY, maxZ);
    }

    private static boolean hasFullZSlice(Set<PositionKey> positions, int y, int minX, int maxX, int z) {
        for (int x = minX; x <= maxX; x++) {
            if (!positions.contains(new PositionKey(x, y, z))) {
                return false;
            }
        }
        return true;
    }

    private static boolean hasFullYSlice(Set<PositionKey> positions, int y, int minX, int maxX, int minZ, int maxZ) {
        for (int x = minX; x <= maxX; x++) {
            for (int z = minZ; z <= maxZ; z++) {
                if (!positions.contains(new PositionKey(x, y, z))) {
                    return false;
                }
            }
        }
        return true;
    }

    private static void removeCuboid(Cuboid cuboid, Set<PositionKey> positions) {
        for (int x = cuboid.minX(); x <= cuboid.maxX(); x++) {
            for (int y = cuboid.minY(); y <= cuboid.maxY(); y++) {
                for (int z = cuboid.minZ(); z <= cuboid.maxZ(); z++) {
                    positions.remove(new PositionKey(x, y, z));
                }
            }
        }
    }

    record Placement(int x, int y, int z, String blockId) {
    }

    record Cuboid(
            String blockId,
            int minX,
            int minY,
            int minZ,
            int maxX,
            int maxY,
            int maxZ
    ) {
        int volume() {
            return (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1);
        }
    }

    record Plan(List<Cuboid> fillCuboids, List<Placement> setBlocks) {
    }

    private record PositionKey(int x, int y, int z) {
    }
}
