-- ============================================================================
-- Supabase schema for the Agentic AI Chatbot
-- Run this ONCE in: Supabase Dashboard -> SQL Editor -> New query
-- ============================================================================

-- 1. Chat message log table (written by agentic_chatbot_backend.py)
create table if not exists public.chat_messages (
    id          bigint generated always as identity primary key,
    thread_id   text not null,
    role        text not null check (role in ('user', 'assistant', 'system')),
    content     text not null,
    created_at  timestamptz not null default now()
);

-- 2. Fast lookup of all messages in one conversation thread
create index if not exists idx_chat_messages_thread_id
    on public.chat_messages (thread_id, created_at);

-- 3. Row Level Security: enable it, then choose ONE policy style below.
alter table public.chat_messages enable row level security;

-- Option A (quick demo / coursework): service-role key bypasses RLS entirely,
-- so with SUPABASE_SERVICE_ROLE_KEY in .env no policy is strictly required.
-- For an anon key, allow full access (NOT for production):
create policy if not exists "anon full access demo"
    on public.chat_messages
    for all
    to anon
    using (true)
    with check (true);

-- Option B (production sketch, if you add auth later):
-- create policy "users read own threads"
--     on public.chat_messages for select
--     using (thread_id = auth.uid()::text);

-- ============================================================================
-- NOTE: LangGraph's persistent checkpoints are stored separately by
-- PostgresSaver via SUPABASE_DB_URL (tables: checkpoints, checkpoint_blobs,
-- checkpoint_writes). Those are auto-created on first run -- nothing to do.
-- ============================================================================
