# Convex Setup

This directory defines the schema and basic functions expected by BrowseCraft:

- `blueprints` table:
  - `name`
  - `world_id`
  - `plan_json`
  - `created_at`
- `sessions` table:
  - `world_id`
  - `session_id`
  - `messages`
  - `created_at`
  - `updated_at`

To deploy:

1. Install Convex CLI.
2. Authenticate with your Convex account.
3. Run `npx convex dev` (or `npx convex deploy`) from the repository root.

Use the resulting deployment URL and access key in backend `.env`:

- `CONVEX_URL`
- `CONVEX_ACCESS_KEY`
