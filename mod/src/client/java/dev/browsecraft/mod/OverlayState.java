package dev.browsecraft.mod;

import net.minecraft.util.math.BlockPos;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

public final class OverlayState {
    private BuildPlan plan;
    private final List<BuildPlacement> remainingPlacements = new ArrayList<>();
    private BlockPos anchor = BlockPos.ORIGIN;
    private int rotationQuarterTurns;
    private boolean previewMode;
    private boolean confirmed;
    private boolean autoAnchorInPreview;

    public synchronized void setPlan(BuildPlan plan) {
        this.plan = plan;
        this.remainingPlacements.clear();
        this.remainingPlacements.addAll(plan.placements());
        this.rotationQuarterTurns = 0;
        this.previewMode = true;
        this.confirmed = false;
        this.autoAnchorInPreview = true;
    }

    public synchronized BlueprintState blueprintState() {
        if (this.plan == null) {
            throw new IllegalStateException("No active plan");
        }
        return new BlueprintState(this.plan, this.anchor, this.rotationQuarterTurns);
    }

    public synchronized void loadBlueprint(BlueprintState blueprintState) {
        setPlan(blueprintState.plan());
        this.anchor = blueprintState.anchor();
        this.rotationQuarterTurns = Math.floorMod(blueprintState.rotationQuarterTurns(), 4);
        this.autoAnchorInPreview = false;
    }

    public synchronized void rotateClockwise() {
        this.rotationQuarterTurns = (this.rotationQuarterTurns + 1) % 4;
    }

    public synchronized void shiftVertical(int deltaY) {
        shift(0, deltaY, 0);
    }

    public synchronized void shift(int deltaX, int deltaY, int deltaZ) {
        this.anchor = this.anchor.add(deltaX, deltaY, deltaZ);
    }

    public synchronized void setAnchor(BlockPos anchor) {
        this.anchor = anchor;
        this.autoAnchorInPreview = false;
    }

    public synchronized void applyInitialPreviewAnchor(BlockPos anchor) {
        if (!this.previewMode || !this.autoAnchorInPreview) {
            return;
        }
        this.anchor = anchor;
    }

    public synchronized void setRotationQuarterTurns(int rotationQuarterTurns) {
        this.rotationQuarterTurns = Math.floorMod(rotationQuarterTurns, 4);
    }

    public synchronized void confirm() {
        this.confirmed = true;
        this.previewMode = false;
    }

    public synchronized void cancel() {
        this.plan = null;
        this.remainingPlacements.clear();
        this.anchor = BlockPos.ORIGIN;
        this.rotationQuarterTurns = 0;
        this.previewMode = false;
        this.confirmed = false;
        this.autoAnchorInPreview = false;
    }

    public synchronized boolean hasPlan() {
        return this.plan != null;
    }

    public synchronized boolean isPreviewMode() {
        return this.previewMode;
    }

    public synchronized boolean isConfirmed() {
        return this.confirmed;
    }

    public synchronized BlockPos anchor() {
        return this.anchor;
    }

    public synchronized int rotationQuarterTurns() {
        return this.rotationQuarterTurns;
    }

    public synchronized int remainingCount() {
        return this.remainingPlacements.size();
    }

    public synchronized List<TransformedPlacement> transformedPlacements() {
        List<TransformedPlacement> transformed = new ArrayList<>(this.remainingPlacements.size());
        for (BuildPlacement placement : this.remainingPlacements) {
            transformed.add(transformPlacement(placement));
        }
        return transformed;
    }

    public synchronized int updateSatisfiedBlocks(Function<BlockPos, String> blockIdLookup) {
        if (!this.confirmed || this.plan == null) {
            return 0;
        }

        int removed = 0;
        Iterator<BuildPlacement> iterator = this.remainingPlacements.iterator();
        while (iterator.hasNext()) {
            BuildPlacement placement = iterator.next();
            TransformedPlacement transformed = transformPlacement(placement);
            String observed = blockIdLookup.apply(transformed.pos());
            if (placement.blockId().equals(observed)) {
                iterator.remove();
                removed++;
            }
        }

        return removed;
    }

    public synchronized int replaceRemainingBlock(String fromBlockId, String toBlockId) {
        int replaced = 0;
        for (int index = 0; index < this.remainingPlacements.size(); index++) {
            BuildPlacement placement = this.remainingPlacements.get(index);
            if (!placement.blockId().equals(fromBlockId)) {
                continue;
            }
            this.remainingPlacements.set(index, new BuildPlacement(
                    placement.dx(),
                    placement.dy(),
                    placement.dz(),
                    toBlockId,
                    placement.blockState()
            ));
            replaced++;
        }
        return replaced;
    }

    public synchronized OverlaySnapshot snapshot() {
        return new OverlaySnapshot(
                this.plan != null,
                this.previewMode,
                this.confirmed,
                this.anchor,
                this.rotationQuarterTurns,
                this.remainingPlacements.size(),
                List.copyOf(transformedPlacements())
        );
    }

    private TransformedPlacement transformPlacement(BuildPlacement placement) {
        int x = placement.dx();
        int z = placement.dz();
        int rotatedX;
        int rotatedZ;

        switch (this.rotationQuarterTurns) {
            case 1 -> {
                rotatedX = -z;
                rotatedZ = x;
            }
            case 2 -> {
                rotatedX = -x;
                rotatedZ = -z;
            }
            case 3 -> {
                rotatedX = z;
                rotatedZ = -x;
            }
            default -> {
                rotatedX = x;
                rotatedZ = z;
            }
        }

        BlockPos transformedPos = this.anchor.add(rotatedX, placement.dy(), rotatedZ);
        return new TransformedPlacement(transformedPos, placement.blockId());
    }

    public record TransformedPlacement(BlockPos pos, String blockId) {}

    public record BlueprintState(BuildPlan plan, BlockPos anchor, int rotationQuarterTurns) {}
}
