package dev.browsecraft.mod;

import net.fabricmc.fabric.api.client.rendering.v1.world.WorldRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.world.WorldRenderEvents;
import net.minecraft.client.render.DrawStyle;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.Box;
import net.minecraft.world.debug.gizmo.GizmoDrawing;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class GhostRenderer {
    private static final float MISSING_MATERIAL_ALPHA_FACTOR = 0.35f;
    private static final float OUTLINE_LINE_WIDTH = 2.0f;

    private final OverlayState overlayState;
    private final Map<String, Integer> colorCache = new HashMap<>();
    private volatile Set<String> availableBlockTypes = Set.of();
    private volatile RenderFrameSnapshot latestFrameSnapshot = new RenderFrameSnapshot(
            RenderMode.NONE,
            0,
            BlockPos.ORIGIN,
            BlockPos.ORIGIN,
            0
    );

    public GhostRenderer(OverlayState overlayState) {
        this.overlayState = overlayState;
    }

    public void register() {
        WorldRenderEvents.BEFORE_DEBUG_RENDER.register(this::onRender);
    }

    public RenderFrameSnapshot latestFrameSnapshot() {
        return latestFrameSnapshot;
    }

    public void setAvailableBlockTypes(Set<String> availableBlockTypes) {
        this.availableBlockTypes = Set.copyOf(availableBlockTypes);
    }

    private void onRender(WorldRenderContext context) {
        List<OverlayState.TransformedPlacement> placements = overlayState.transformedPlacements();
        if (placements.isEmpty()) {
            latestFrameSnapshot = new RenderFrameSnapshot(RenderMode.NONE, 0, BlockPos.ORIGIN, BlockPos.ORIGIN, 0);
            return;
        }

        boolean drawOutlines = placements.size() <= 1500;

        BlockPos min = placements.getFirst().pos();
        BlockPos max = placements.getFirst().pos();
        Set<String> missingBlockTypes = new HashSet<>();

        for (OverlayState.TransformedPlacement placement : placements) {
            BlockPos pos = placement.pos();
            min = new BlockPos(
                    Math.min(min.getX(), pos.getX()),
                    Math.min(min.getY(), pos.getY()),
                    Math.min(min.getZ(), pos.getZ())
            );
            max = new BlockPos(
                    Math.max(max.getX(), pos.getX()),
                    Math.max(max.getY(), pos.getY()),
                    Math.max(max.getZ(), pos.getZ())
            );

            float alphaFactor = alphaFactorForBlock(placement.blockId());
            if (alphaFactor < 1.0f) {
                missingBlockTypes.add(placement.blockId());
            }
            if (drawOutlines) {
                int color = applyAlpha(colorForBlock(placement.blockId()), alphaFactor);
                GizmoDrawing.box(pos, DrawStyle.stroked(color, OUTLINE_LINE_WIDTH));
            }
        }

        if (!drawOutlines) {
            Box bounds = new Box(
                    min.getX(),
                    min.getY(),
                    min.getZ(),
                    max.getX() + 1.0,
                    max.getY() + 1.0,
                    max.getZ() + 1.0
            );
            GizmoDrawing.box(bounds, DrawStyle.stroked(0xFF66B3FF, OUTLINE_LINE_WIDTH));
            latestFrameSnapshot = new RenderFrameSnapshot(
                    RenderMode.BOUNDING_BOX,
                    placements.size(),
                    min,
                    max,
                    missingBlockTypes.size()
            );
            return;
        }

        latestFrameSnapshot = new RenderFrameSnapshot(
                RenderMode.OUTLINES,
                placements.size(),
                min,
                max,
                missingBlockTypes.size()
        );
    }

    private int colorForBlock(String blockId) {
        return colorCache.computeIfAbsent(blockId, this::resolveColor);
    }

    private float alphaFactorForBlock(String blockId) {
        if (availableBlockTypes.contains(blockId)) {
            return 1.0f;
        }
        return MISSING_MATERIAL_ALPHA_FACTOR;
    }

    private int applyAlpha(int color, float alphaFactor) {
        int alpha = Math.clamp(Math.round(255.0f * alphaFactor), 0, 255);
        return (alpha << 24) | (color & 0x00FF_FFFF);
    }

    private int resolveColor(String blockId) {
        String lower = blockId.toLowerCase(Locale.ROOT);

        if (matchesRedstone(lower)) {
            return 0xFFE65A5A;
        }
        if (matchesWood(lower)) {
            return 0xFFC98B4B;
        }
        if (matchesMetal(lower)) {
            return 0xFFD0DBE5;
        }
        if (matchesStone(lower)) {
            return 0xFF9AA2AD;
        }
        if (matchesGlass(lower)) {
            return 0xFF81D8FF;
        }
        if (matchesNature(lower)) {
            return 0xFF76BC7F;
        }
        return 0xFF66B3FF;
    }

    private boolean matchesRedstone(String value) {
        return value.contains("redstone")
                || value.contains("repeater")
                || value.contains("comparator")
                || value.contains("observer")
                || value.contains("piston")
                || value.contains("hopper")
                || value.contains("dispenser")
                || value.contains("dropper")
                || value.contains("lever");
    }

    private boolean matchesWood(String value) {
        return value.contains("oak_")
                || value.contains("spruce_")
                || value.contains("birch_")
                || value.contains("jungle_")
                || value.contains("acacia_")
                || value.contains("dark_oak_")
                || value.contains("mangrove_")
                || value.contains("cherry_")
                || value.contains("pale_oak_")
                || value.contains("bamboo_")
                || value.contains("crimson_")
                || value.contains("warped_")
                || value.contains("planks")
                || value.contains("log")
                || value.contains("wood")
                || value.contains("hyphae")
                || value.contains("stem");
    }

    private boolean matchesMetal(String value) {
        return value.contains("iron")
                || value.contains("gold")
                || value.contains("copper")
                || value.contains("netherite")
                || value.contains("anvil")
                || value.contains("chain")
                || value.contains("cauldron")
                || value.contains("rail");
    }

    private boolean matchesStone(String value) {
        return value.contains("stone")
                || value.contains("cobblestone")
                || value.contains("deepslate")
                || value.contains("tuff")
                || value.contains("andesite")
                || value.contains("diorite")
                || value.contains("granite")
                || value.contains("basalt");
    }

    private boolean matchesGlass(String value) {
        return value.contains("glass") || value.contains("pane") || value.contains("ice");
    }

    private boolean matchesNature(String value) {
        return value.contains("dirt")
                || value.contains("grass")
                || value.contains("moss")
                || value.contains("mud")
                || value.contains("clay")
                || value.contains("sand")
                || value.contains("gravel")
                || value.contains("snow")
                || value.contains("leaf")
                || value.contains("flower")
                || value.contains("vine")
                || value.contains("mushroom")
                || value.contains("farmland");
    }

    public enum RenderMode {
        NONE,
        OUTLINES,
        BOUNDING_BOX
    }

    public record RenderFrameSnapshot(
            RenderMode mode,
            int placementCount,
            BlockPos minCorner,
            BlockPos maxCorner,
            int missingMaterialTypes
    ) {}
}
