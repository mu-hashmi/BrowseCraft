import { mutation, query } from "./_generated/server";
import { v } from "convex/values";

export const upsert = mutation({
  args: {
    world_id: v.string(),
    session_id: v.string(),
    messages: v.array(v.any()),
    created_at: v.number(),
    updated_at: v.number(),
  },
  handler: async (ctx, args) => {
    const existing = await ctx.db
      .query("sessions")
      .withIndex("by_world_id_session_id", (q) =>
        q.eq("world_id", args.world_id).eq("session_id", args.session_id),
      )
      .first();

    if (existing) {
      await ctx.db.patch(existing._id, {
        messages: args.messages,
        updated_at: args.updated_at,
      });
      return existing._id;
    }

    return await ctx.db.insert("sessions", args);
  },
});

export const get = query({
  args: {
    world_id: v.string(),
    session_id: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("sessions")
      .withIndex("by_world_id_session_id", (q) =>
        q.eq("world_id", args.world_id).eq("session_id", args.session_id),
      )
      .first();
  },
});

export const listByWorld = query({
  args: {
    world_id: v.string(),
  },
  handler: async (ctx, args) => {
    return await ctx.db
      .query("sessions")
      .withIndex("by_world_id_session_id", (q) => q.eq("world_id", args.world_id))
      .collect();
  },
});
