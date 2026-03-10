package dev.browsecraft.mod;

import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WorldIdResolverTest {

    @Test
    void singleplayerWorldIdIsStableForEquivalentPaths() {
        String first = WorldIdResolver.singleplayerWorldId(Path.of("saves/world-a"));
        String second = WorldIdResolver.singleplayerWorldId(Path.of("saves/./world-a"));

        assertEquals(first, second);
        assertTrue(first.startsWith("sp:"));
    }

    @Test
    void singleplayerWorldIdChangesAcrossDifferentWorlds() {
        String first = WorldIdResolver.singleplayerWorldId(Path.of("saves/world-a"));
        String second = WorldIdResolver.singleplayerWorldId(Path.of("saves/world-b"));

        assertNotEquals(first, second);
    }

    @Test
    void multiplayerWorldIdNormalizesServerAddress() {
        String first = WorldIdResolver.multiplayerWorldId("EXAMPLE.com:25565");
        String second = WorldIdResolver.multiplayerWorldId("example.com:25565 ");

        assertEquals(first, second);
        assertTrue(first.startsWith("mp:"));
    }

    @Test
    void multiplayerWorldIdChangesAcrossServers() {
        String first = WorldIdResolver.multiplayerWorldId("example.com:25565");
        String second = WorldIdResolver.multiplayerWorldId("example.com:25566");

        assertNotEquals(first, second);
    }
}
