# Chaos Sail


## Installation

```bash
# download the file and name it chaos_deck.py
chmod +x chaos_deck.py
# (optional) put it on your PATH
mv chaos-sail.py ~/.local/bin/chaos-sail.py
```

### Environment variables

* `CHAOS_ROOT` — root directory (default: `~/ChaosSail`).
* `EDITOR` / `VISUAL` — preferred editor (Linux/macOS). On Windows the fallback is `notepad`.

Example:

```bash
export CHAOS_ROOT=~/ChaosSail
export EDITOR="code -w"    # VS Code
```

---

## Quick start

```bash
# 1) Initialize (creates the root and journal.md)
chaos_deck.py init

# 2) Spark — a short thought (one line)
chaos_deck.py spark "riff in Em, dotted 8th delay #music"

# 3) Flow — open the editor and work 10–40 minutes
chaos_deck.py flow "north-wind-chorus"

# 4) Trace — attach an artifact (a file)
chaos_deck.py trace ~/music/north_wind_demo.mp3 "first take #music"
```

---

## Commands

| Command              | What it does                                                                                    | Example                                                 |
| -------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `init`               | Creates the base structure and the journal                                                      | `chaos_deck.py init`                                    |
| `spark [text...]`    | Short note (≤ 5 min). If no text is provided, it opens your editor and saves a file in `spark/` | `chaos_deck.py spark "idea: Em→G→D"`                    |
| `flow [name]`        | Opens the editor for a focused chunk of work (file goes to `flow/`)                             | `chaos_deck.py flow "docker-multistage"`                |
| `trace <file> [msg]` | Copies an artifact into `trace/` and appends one line to the journal                            | `chaos_deck.py trace ~/shots/k8s_pods.png "green #k8s"` |
| `roll`               | Rolls a d6 to suggest where to go next (music/languages/hardware/text/research/wildcard)        | `chaos_deck.py roll`                                    |

Common flag:

* `--root PATH` — use an alternative root instead of `CHAOS_ROOT`.

---

## Folder layout

```
~/ChaosSail/
  journal.md
  2025/
    08/
      2025-08-25/
        spark/
          091502.txt
        flow/
          101244-north-wind-chorus.md
        trace/
          112533-north_wind_demo.mp3
```

* **Each day** gets its own `YYYY/MM/YYYY-MM-DD/` folder.
* Inside are three baskets: `spark/`, `flow/`, and `trace/`.

---

## Journal (`journal.md`)

Format: **one line per action**.

```text
2025-08-25 | spark: 091502.txt | #music
2025-08-25 | flow: 101244-north-wind-chorus.md
2025-08-25 | trace: 112533-north_wind_demo.mp3 | first take #music
```

Tips:

* Add **hashtags** in `trace`/`spark` comments — they make searching later trivial.
* Don’t turn the journal into a report: one line, then back to work.

---

## Example scenarios

### Music

```bash
chaos_deck.py spark "hook in Em #music"
chaos_deck.py flow  "northern-verse"
chaos_deck.py trace ~/bounces/northern_verse_v1.mp3 "sketch #music"
```

### Languages

```bash
chaos_deck.py spark "AM: inch bes es? #armenian #languages"
chaos_deck.py flow  "armenian-mini-monologue"
chaos_deck.py trace ~/voice/arm_2025-08-25.ogg "no cuts #languages"
```

### Hardware / RPi

```bash
chaos_deck.py spark "GPIO smoke plan #rpi #hardware"
chaos_deck.py flow  "led-fade-proto"
chaos_deck.py trace ~/photos/led_fade_breadboard.jpg "ok #hardware"
```

---

## Philosophy (why it’s this simple)

* **No bureaucracy.** Exactly three actions: Spark → Flow → Trace.
* **Trace over status.** If there’s no artifact, it’s as if you didn’t do it.
* **A song instead of a timer.** One music track ≈ one flow session.

---

## FAQ

**Where do artifacts live?**
The script copies the provided file into `trace/` — the original stays where it is. You can back up “traces” with git or any cloud.

**What if I don’t type anything for `spark`?**
The command opens your `$EDITOR`; even an empty file is created and gets logged.

**Can I change the editor?**
Yes: set `EDITOR`/`VISUAL`. On Windows the fallback is `vim`, sorry :-P.

---

## Micro‑hacks

* If you’re stuck — run `roll` and do a 60‑second *spark*.
* Keep ≤ 2 “big obsessions” active; park everything else.
* Prefer **micro‑releases**: five rough drafts in `trace/` beat one “perfect” thing a month from now.

---

## License

MIT.
