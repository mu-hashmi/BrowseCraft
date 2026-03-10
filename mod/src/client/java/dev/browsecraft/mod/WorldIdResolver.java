package dev.browsecraft.mod;

import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ServerInfo;
import net.minecraft.util.WorldSavePath;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

public final class WorldIdResolver {
    private static final char[] HEX = "0123456789abcdef".toCharArray();

    public String resolve(MinecraftClient client) {
        if (client.world == null) {
            throw new IllegalStateException("world_id requires an active world");
        }

        if (client.getServer() != null) {
            Path savePath = client.getServer().getSavePath(WorldSavePath.ROOT);
            return singleplayerWorldId(savePath);
        }

        ServerInfo serverInfo = client.getCurrentServerEntry();
        if (serverInfo == null) {
            throw new IllegalStateException("world_id requires an active server entry");
        }
        return multiplayerWorldId(serverInfo.address);
    }

    static String singleplayerWorldId(Path savePath) {
        String normalized = savePath.toAbsolutePath().normalize().toString().replace('\\', '/');
        return "sp:" + sha256("sp|" + normalized);
    }

    static String multiplayerWorldId(String serverAddress) {
        String normalized = serverAddress.trim().toLowerCase(Locale.ROOT);
        return "mp:" + sha256("mp|" + normalized);
    }

    private static String sha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            char[] hex = new char[bytes.length * 2];
            int index = 0;
            for (byte value : bytes) {
                int unsigned = value & 0xFF;
                hex[index++] = HEX[(unsigned >>> 4) & 0x0F];
                hex[index++] = HEX[unsigned & 0x0F];
            }
            return new String(hex);
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 unavailable", error);
        }
    }
}
