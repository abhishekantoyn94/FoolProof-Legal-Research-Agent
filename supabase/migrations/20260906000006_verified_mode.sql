-- Adds 'verified' research mode: DEEP-style multi-round agent loop plus a
-- web-search evidence channel against an allowlist of primary Indian legal
-- sources (indiankanoon.org, sci.gov.in, gov.in/nic.in, etc.), with every
-- piece of evidence tagged by source authority (user_document/primary/secondary)
-- so synthesis/challenge can weight a primary source over a secondary one
-- when they conflict, rather than treating all evidence as equally authoritative.

alter table research_sessions drop constraint research_sessions_research_mode_check;
alter table research_sessions add constraint research_sessions_research_mode_check
    check (research_mode in ('simple', 'quick', 'standard', 'deep', 'verified'));
