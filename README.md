# Jarvis

A personal push-to-talk voice assistant for Windows. Hold a key, speak, and hear a calm British butler answer.

> I have no real coding background beyond the basics; Claude Code wrote this under my direction. It works for me on one Windows 11 laptop, so expect rough edges. I'd like to port it to a Raspberry Pi eventually. Suggestions (design, safety, optimisation) are very welcome.

Rough shape: wake word ("Hey Jarvis") or push-to-talk, then cloud speech-to-text, an LLM with tools (email, calendar, to-dos, timers, web search, a gym tracker, a mobile-games news digest), then a local British voice with a "voice in the walls" effect. A sci-fi HUD (adapted from A.L.I.C.E.) shows what it's doing, and there's a small round glass widget for the desktop.

## Setup on Windows

You need Python 3.10 or newer and Git. Run these in PowerShell from this folder.

1. **Create the project's own Python environment and install** (about 200 MB, one time):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
   ```

2. **Add your API key.** Copy `.env.example` to `.env`, open `.env` in Notepad and paste your key after `GROQ_API_KEY=`.

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Get a free Groq key at https://console.groq.com/keys. `.env` is gitignored, so it never ends up in the repo.

   **Free tiers may use your prompts for training. Only use dummy data with Groq** - nothing from your email, meal plan or the shop.

3. **Download the voices.** The main voice is Kokoro, about 120 MB: the model plus the voice pack.

   ```powershell
   New-Item -ItemType Directory -Force models\kokoro
   Invoke-WebRequest https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx -OutFile models\kokoro\kokoro-v1.0.int8.onnx
   Invoke-WebRequest https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -OutFile models\kokoro\voices-v1.0.bin
   ```

   The fallback voice is Piper, about 60 MB. Jarvis uses it if Kokoro won't load:

   ```powershell
   .\.venv\Scripts\python.exe -m piper.download_voices --data-dir models\piper en_GB-northern_english_male-medium
   ```

   If neither loads, Jarvis falls back to the Windows voice. That's US English "David" unless you add the English (UK) speech pack.

## Running

```powershell
.\.venv\Scripts\python.exe -m jarvis                  # voice: hold F9, speak, release (opens the HUD)
.\.venv\Scripts\python.exe -m jarvis --no-hud         # voice, without the screen
.\.venv\Scripts\python.exe -m jarvis --no-wake        # voice, push-to-talk only (no "hey Jarvis")
.\.venv\Scripts\python.exe -m jarvis --text           # type instead of talk
.\.venv\Scripts\python.exe -m jarvis --text --hud     # type, and watch the screen
.\.venv\Scripts\python.exe -m jarvis --text --speak   # type, hear the reply
.\.venv\Scripts\python.exe -m jarvis --list-devices   # microphones and speakers
.\.venv\Scripts\python.exe -m jarvis --list-voices    # installed voices
.\.venv\Scripts\python.exe -m jarvis --google-login personal   # sign a Google account in (see below)
.\.venv\Scripts\python.exe -m pytest                  # run the tests
```

Say **"Hey Jarvis"** and ask, or hold F9. Press F9 while Jarvis is talking to cut him off. Ctrl+C quits.

## One click: pin Jarvis to the taskbar

```powershell
.\.venv\Scripts\python.exe -m jarvis --install-shortcut
```

This adds **Jarvis** to the Start menu. Press the Windows key, type *Jarvis*, right-click it and choose **Pin to taskbar** (Windows doesn't let programs pin themselves). From then on one click starts Jarvis with the HUD in its own window and no console; **closing the HUD window quits Jarvis**; clicking the icon while he's running brings the window to the front. If something goes wrong at start-up you'll get a message box; details are in `logs\jarvis.log` (and `logs\hud-window.log` for the window).

## The HUD (screen)

Voice mode opens the HUD in **its own window** (no browser needed; it uses Microsoft Edge WebView2, which comes with Windows 11). Closing Jarvis closes the window. If you close the window, Jarvis keeps running; the HUD is still at http://127.0.0.1:8765/ in any browser. It shows:

- **The ring**, which changes with Jarvis's state: calm (standby), bright (listening), amber (processing) and pulsing (responding). It follows your voice while you speak and Jarvis's voice while he answers.
- **Your words and his reply.** The reply is typed out as he says it.
- **A clock and the date** in the centre while he's on standby.
- **Meals** (top right): today's meals from the meal plan sheets, current meal highlighted, switching between plans every 12 seconds. Refreshed every 10 minutes and just after midnight.
- **Timers** (left): your next two timers counting down.
- **Digest** (bottom right): new mobile games news since the last digest, the top headlines.
- **Calendar** (bottom left): what's left of today, or tomorrow once today's done, across all your calendars.

**Hold the ring** with the mouse, or with a finger on a touchscreen, to talk. It works the same as holding F9.

For a cleaner full-screen window with no browser bars, Edge has a kiosk mode. Press Alt+F4 to leave it:

```powershell
Start-Process msedge -ArgumentList '--kiosk','http://127.0.0.1:8765/','--edge-kiosk-type=fullscreen','--no-first-run'
```

The page is only reachable from this computer, and it refuses connections from other websites. Settings are under `[hud]` in `config.toml`: `enabled`, `port`, `window` (`"app"` for its own window, `"browser"` for a browser tab, `"none"`) and `fullscreen`. The text you say is shown on screen but never written to disk.

## Privacy: what stays local and what doesn't

This is **not** a local-LLM setup.

- **Stays on the laptop:** the wake word (the microphone is only listened to locally until it hears "Hey Jarvis"), the voice, and the HUD. No audio or transcripts are written to disk; the logs hold timings, token counts and tool names only.
- **Goes to the cloud:** speech-to-text (Groq Whisper by default; `[stt] provider = "local"` runs offline) and the LLM. When a question needs data, the relevant snippet goes to the LLM too. For email that is sender, subject, time and Gmail's short preview, never whole messages, and each account has a Gmail filter limiting what is visible.
- **Provider:** `groq` is the default here for testing with dummy data only; free tiers may use prompts for training. For real use, switch `[llm] provider` to `anthropic` (Claude Haiku) and read your provider's data terms first.

## Settings

`JARVIS_TZ` in `.env` sets your time zone (an IANA name such as `America/New_York`; defaults to UTC).

Everything non-secret is in `config.toml`:

- `[llm] provider`: `groq`, `openrouter` or `anthropic`. Switching provider is this one line (or `--provider anthropic` for a single run).
- `max_tokens` (default 200) caps each reply; `max_turns_per_session` stops runaway use.
- `[stt] provider`: `groq` (cloud Whisper) or `local` (offline; first run `pip install -e ".[local]"`).
- `[tts] provider`: `kokoro` (default, most natural), `piper` (fastest) or `sapi`. `kokoro_voice` picks the voice: `bm_daniel` by default, or `bm_george`, `bm_lewis` or `bm_fable`. `kokoro_speed` sets the pace.
- `[tts] effect`: `film` gives the processed "voice in the walls" sound; `none` turns it off.
- `[audio] push_to_talk_key`: default `f9`.

The persona is plain text in `jarvis/persona/jarvis.md`; edit it freely.

## Google accounts (email)

Jarvis can read your recent email, **read-only**, across several Google accounts. He sees the sender, subject, time and Gmail's short preview, never whole messages, and he can't send, delete or change anything. Ask things like "any new emails?", "anything from Etsy today?" or "anything unread on the work account?".

**One-time setup (about 10 minutes, all clicking):**

1. Go to https://console.cloud.google.com/ and sign in with your personal Google account. Accept the terms if asked.
2. Click the project picker at the top, then **New project**. Name it `Jarvis`, click **Create**, and make sure it's selected.
3. Menu (☰) → **APIs & Services** → **Library**. Search for and **Enable** each of: **Gmail API**, **Google Calendar API**, **Google Sheets API**.
4. Menu → **Google Auth Platform** (it may be called "OAuth consent screen") → **Get started**. App name `Jarvis (personal)`, your email as support email, Audience **External**, your email as contact. Agree, then **Create**.
5. **Audience** → **Test users** → **Add users**: add every Google account Jarvis will read. Leave the status as **Testing**. (Publishing needs a homepage and privacy policy on your own domain. In Testing, Google signs each account out after 7 days; `--google-login all` signs them all back in.)
6. **Clients** → **Create client**. Application type **Desktop app**, name `Jarvis laptop`, **Create**, then **Download JSON**.
7. Make a folder `tokens` inside the project folder and save the file there as `client_secret.json`. The folder is gitignored; never share or commit that file.

**Your accounts** are listed in `config.local.toml` (also gitignored; `config.local.example.toml` shows the format). Each has a name, its address, what it's for, which services it may use, and a Gmail filter (`gmail_query`) that decides which mail Jarvis can see, for example `in:inbox category:primary` for the Primary tab only. He can narrow a search but never look outside that filter.

**Sign each account in once:**

```powershell
.\.venv\Scripts\python.exe -m jarvis --google-login all        # every account, one after another (weekly)
.\.venv\Scripts\python.exe -m jarvis --google-login personal
.\.venv\Scripts\python.exe -m jarvis --google-login shop
.\.venv\Scripts\python.exe -m jarvis --google-login work
.\.venv\Scripts\python.exe -m jarvis --google-status    # checks each one
```

A browser opens. Pick the matching account. Google warns that the app isn't verified: click **Continue** (or **Advanced**, then **Go to Jarvis**). That warning appears because it's your own unpublished app. Leave all the boxes ticked, then **Continue**. Jarvis checks you picked the right account before saving anything.

To revoke access at any time: https://myaccount.google.com/permissions → Jarvis → remove access. Or delete the account's file in `tokens\`. Settings are under `[google]` and `[gmail]` in `config.toml`.

A work (Google Workspace) account may be blocked by its administrator from signing in to apps like this one; if so, `--google-login work` will say so.

## Calendar

Jarvis reads all three accounts' Google calendars (plus UK holidays) and can **add, move and delete events by voice** on the personal and the shop calendars. "What's on this week?", "Put the Spitalfields market in for Saturday, nine till five, on the Shop calendar", "Add a call with my editor on Tuesday at 3", "Move that to 4", "Delete that".

- **He always asks first.** He reads the change back ("...to the shop calendar. Shall I add it?") and only makes it if your *next* words are a plain yes ("yes", "go ahead", "do it"). Anything else, including "yes, but make it 4", changes nothing (he proposes the new version and asks again). This check is in Jarvis's code, not left to the AI, so text in an email can never count as your yes.
- He can only move or delete events he added himself; anything you or others put in is read-only to him. The work calendar is read-only.
- Events he adds say "Added by Jarvis." in their description, and show on your phone through Google Calendar as usual.
- **Embargoes:** "Any embargoes coming up?" searches the last two weeks of email for "embargo" and reads him only the sentences about it; he can then offer to add each one. Once a day he mentions new ones.

Adding events needs one more Google permission, so after updating, sign the personal and Shop accounts in again (`--google-login personal`, `--google-login shop`); until then he can read but will say the account needs signing in again. Which accounts can be changed is the `calendar_write` service in `config.local.toml`.

## To-do list

Your to-do list lives in **Google Tasks** on the personal account, so it's also in the Google Tasks app and Google Calendar on your phone. "Add to my list: pitch the cosy games feature, due Friday", "add order more pasta", "add file the review by 5pm Monday", "what's on my list?", "what's due this week?", "mark the pasta one as done", "move the pitch to Monday", "undo that", "delete the pitch".

- Adding, ticking off and moving happen straight away and are read back; **"undo that"** reverses the last one. **Deleting** asks first and needs your yes.
- A deadline with a time also gets a spoken reminder an hour before (Google Tasks itself only keeps the date; the time is written in the task's notes).
- Once a day he mentions anything due today or overdue. The HUD's **To do** card (right) shows the next three.

Setup, once: enable the Tasks API in the Google Cloud project (https://console.cloud.google.com/apis/library/tasks.googleapis.com, Enable), then `--google-login personal` again.

## "Hey Jarvis" (hands-free)

Say "Hey Jarvis" and then your question in one go ("Hey Jarvis, what's on today?"), or say "Hey Jarvis", wait for "Yes, sir?", then ask. He stops recording after about a second of silence. The microphone is listened to **only on this laptop** by a small wake word model (openWakeWord, about 4% of one CPU core); nothing is sent anywhere until he hears the phrase. Detection pauses while he's talking, so his own voice can't wake him; F9 still interrupts and push-to-talk still works.

The models (about 4 MB, from the project's official GitHub release) go in `models\openwakeword`:

```powershell
mkdir models\openwakeword
foreach ($f in "melspectrogram.onnx","embedding_model.onnx","hey_jarvis_v0.1.onnx") { Invoke-WebRequest "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/$f" -OutFile "models\openwakeword\$f" }
```

Settings in `config.toml` under `[audio]`: `wake_word` (on/off) and `wake_threshold` (0 to 1; raise it if he wakes up to the television, lower it if he misses you). Say "*Hey* Jarvis": plain "Jarvis" is often missed.

## Timers and reminders

"Set a 40-minute timer for the pasta", "remind me at 6 to post the parcels", "remind me tomorrow at 9 to chase the invoice", "what timers have I got?", "cancel the pasta timer". When one's due he chimes and says so (after finishing any sentence). They're kept in `data\timers.json`, so they survive a restart; anything that fell due while he was off is mentioned when he starts.

## Meal plans

Jarvis reads the dietitian's weekly plans from Google Sheets (read-only) through the personal account. Ask "what's my dinner today?", "what's Sam having for lunch tomorrow?" or "what's my breakfast on Monday?". He finds the date in whichever week tab holds it, so new weeks and renamed meals just work. Only that one day's meals go to the AI model; recipes, grocery lists and the tracker tabs are never read out. If a week hasn't been added yet, he says so.

The plans are listed in `config.local.toml` under `[meal_plan.plans.<name>]` (sheet id, the Google account it's shared with, and who it's for). Settings are under `[meal_plan]` in `config.toml`; `hud_card = false` hides the HUD card.

## Mobile games digest

While Jarvis runs, he checks public feeds in the background: GamesPress (hourly; only items that mention iOS, Android or mobile), Pokémon GO's news page, Gamezebo, Droid Gamers and mobilegamer.biz (every few hours). Guides, codes and puzzle answers are skipped. You can also add a "reference" feed (`role = "covered"` in `config.toml`): it is read only to mark stories it has already covered, and those go to the end of the digest.

- The first time you speak to him each day, he mentions how many new items there are and offers to run through them.
- "What's new in mobile games?" or "give me the games digest" reads everything since the last digest. "Any Pokémon GO news this week?" or "anything about Monopoly Go today?" narrows it down.
- What's been seen is kept in `data/games.json` for 30 days, so nothing is reported twice.

Sources and intervals are under `[games.sources.*]` in `config.toml`; set `enabled = false` on one to drop it. GamesPress's feed only holds its latest 20 items (a few hours on a weekday), so items published while the laptop is off for longer are missed.

### Using Claude

1. Create an account at https://console.anthropic.com/ and **set a monthly spend limit before adding credit**.
2. Put the key in `.env` as `ANTHROPIC_API_KEY=...`.
3. Set `provider = "anthropic"` in `config.toml`.

Why switch: Groq's free tier allows 8,000 tokens a minute, and with all the tools each question now uses about 1,600 tokens (3,000 if it fetches data), so a few quick questions in a row make Jarvis wait (he says he's being rate-limited, waits as long as Groq asks, up to 12 seconds). Claude Haiku costs about a twentieth of a penny per question at that size. The default model is `claude-haiku-5-5` with thinking off and low effort, which keeps replies fast and cheap.

## Logs

`logs/turns-YYYY-MM-DD.jsonl` has one line per turn: timings (record, transcribe, first token, first audio), token counts, estimated cost, and the names of any tools used (never what they returned). `logs/jarvis.log` has warnings and errors. **No audio and no transcripts are ever written to disk.**

## Licences

- The HUD's ring builders, voice "spoke burst" and screen-fitting approach are adapted from [A.L.I.C.E.](https://github.com/texas0418/ALICE) (MIT, © 2026 Simon Shih). The notice is kept at the top of `jarvis/hud/static/index.html`.
- The Kokoro-82M voice model is Apache-2.0. `kokoro-onnx` is MIT. It bundles espeak-ng, which is GPL-3.0; that's fine for personal use.
- Piper (`piper-tts`) is GPL-3.0. That's fine for personal use; it would matter only if this project were distributed.
- Each Piper voice has its own licence, listed in its model card on Hugging Face.
- "Jarvis" is a character from Marvel's Iron Man. This is a fan-style personal project, unaffiliated with and not endorsed by Marvel or Disney.
- The code in this repository is MIT licensed (see LICENSE). Dependencies keep their own licences, including GPL ones (Piper, espeak-ng via Kokoro) - check them before redistributing a bundle.
