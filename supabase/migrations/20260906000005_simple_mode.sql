-- Adds a 'simple' research mode: single-pass semantic+keyword search with one
-- cross-check synthesis call, no multi-round agent loop. Kept alongside the
-- existing quick/standard/deep modes rather than replacing them.

alter table research_sessions drop constraint research_sessions_research_mode_check;
alter table research_sessions add constraint research_sessions_research_mode_check
    check (research_mode in ('simple', 'quick', 'standard', 'deep'));
