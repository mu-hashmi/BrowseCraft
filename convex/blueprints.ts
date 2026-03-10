import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const upsert = mutation({
  args: {
    name: v.string(),
    world_id: v.string(),
    plan_json: v.any(),
    created_at: v.number(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("blueprints")
      .withIndex("by_world_id_name", (q) =>
        q.eq("world_id", args.world_id).eq("name", args.name),
      )
      .first();

    if (existing) {
      await ctx.db.patch(existing._id, {
        plan_json: args.plan_json,
        created_at: args.created_at,
      });
      return existing._id;
    }

    return await ctx.db.insert("blueprints", args);
  },
});

export const get = query({
  args: {
    world_id: v.string(),
    name: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("blueprints")
      .withIndex("by_world_id_name", (q) =>
        q.eq("world_id", args.world_id).eq("name", args.name),
      )
      .first();
  },
});

export const list = query({
  args: {
    world_id: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("blueprints")
      .withIndex("by_world_id_name", (q) => q.eq("world_id", args.world_id))
      .collect();
  },
});
