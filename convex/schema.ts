import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

export default defineSchema({
  blueprints: defineTable({
    name: v.string(),
    world_id: v.string(),
    plan_json: v.any(),
    created_at: v.number(),
  }).index("by_world_id_name", ["world_id", "name"]),
  sessions: defineTable({
    world_id: v.string(),
    session_id: v.string(),
    messages: v.array(v.any()),
    created_at: v.number(),
    updated_at: v.number(),
  }).index("by_world_id_session_id", ["world_id", "session_id"]),
});
