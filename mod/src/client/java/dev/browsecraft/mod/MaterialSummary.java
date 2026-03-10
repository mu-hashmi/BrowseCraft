package dev.browsecraft.mod;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class MaterialSummary {
    private MaterialSummary() {}

    public static List<MaterialCount> aggregate(List<OverlayState.TransformedPlacement> placements) {
        Map<String, Integer> counts = new HashMap<>();
        for (OverlayState.TransformedPlacement placement : placements) {
            counts.merge(placement.blockId(), 1, Integer::sum);
        }
        List<MaterialCount> result = new ArrayList<>(counts.size());
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            result.add(new MaterialCount(entry.getKey(), entry.getValue()));
        }
        result.sort(Comparator
                .comparingInt(MaterialCount::count).reversed()
                .thenComparing(MaterialCount::blockId));
        return List.copyOf(result);
    }

    public static List<MaterialDelta> compareWithInventory(List<MaterialCount> needed, Map<String, Integer> inventory) {
        List<MaterialDelta> result = new ArrayList<>(needed.size());
        for (MaterialCount count : needed) {
            int inInventory = inventory.getOrDefault(count.blockId(), 0);
            int missing = Math.max(0, count.count() - inInventory);
            result.add(new MaterialDelta(count.blockId(), count.count(), inInventory, missing));
        }
        result.sort(Comparator
                .comparingInt(MaterialDelta::needed).reversed()
                .thenComparing(MaterialDelta::blockId));
        return List.copyOf(result);
    }

    public record MaterialCount(String blockId, int count) {}

    public record MaterialDelta(String blockId, int needed, int inInventory, int missing) {}
}
