-- Phase 5: columns needed for the research agent to be interruptible/resumable
-- (master prompt section 36). `agent_state` is the fine-grained state-machine
-- position (finer than `status`); `state_data` is the accumulated working
-- memory (plan, evidence seen so far, findings, iteration count) needed to
-- resume mid-loop after a pause or crash. `stop_requested` lets an external
-- caller ask a running session to pause at the next safe checkpoint rather
-- than killing the process.

alter table research_sessions
    add column agent_state text not null default 'idle',
    add column state_data jsonb not null default '{}'::jsonb,
    add column stop_requested boolean not null default false;
