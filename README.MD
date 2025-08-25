# Chaos Deck v2 — ultra-light creative cadence toolkit (Python 3, single file)

## Commands (subcommands):
  init                    Create base structure in CHAOS_ROOT (default ~/ChaosDeck)
  spark [text...]         Drop a tiny note (<=5 min idea) into today's `spark/`
  flow  [name]            Open editor for a focused chunk in today's `flow/`
  trace <file> [msg]      Copy an artifact into today's `trace/` and log to journal
  roll                    Roll a d6 to pick a category when you're stuck
  stats-all [opts]        Overall stats (streaks, counts, top tags, ratios)
  stats-tags [tags..]     Stats filtered by tags (supports prefix matching)
  tags                    List known tags (by frequency)
  install-completion      Install simple bash completion for `stats-tags`
  __complete              Internal: completion endpoint (do not call manually)

## Environment:
  CHAOS_ROOT              Root directory (default: ~/ChaosDeck)
  EDITOR / VISUAL         Preferred editor for `flow`/empty `spark`

## Folders for each day (auto-created on demand):
  YYYY/MM/YYYY-MM-DD/{spark,flow,trace}

## Journal:
  $CHAOS_ROOT/journal.md — one line per action (keep it short)

### No dependencies beyond the standard library.
