.PHONY: help refresh refresh-data project predict odds docker-refresh docker-live live-refresh docker-live-up docker-live-logs docker-live-refresh docker-scheduler-up docker-scheduler-logs docker-scheduler-down schedule-on schedule-off schedule-status odds-schedule-on odds-schedule-off odds-schedule-status live-schedule-on live-schedule-off live-schedule-status

YEAR ?= $(shell date +%Y)
PYTHON ?= .venv/bin/python
LAUNCHD_LABEL := com.dmatten.mlb-ev.refresh
LAUNCHD_PLIST := $(HOME)/Library/LaunchAgents/$(LAUNCHD_LABEL).plist

help:
	@echo "MLB EV pipeline — common commands"
	@echo ""
	@echo "  make refresh          — run the full daily pipeline (data + features + schedule + predict)"
	@echo "  make refresh-data     — refresh data only (no schedule/predict)"
	@echo "  make project          — pull tonight's schedule + project lineups + predict"
	@echo "  make predict          — alias for 'make project' (kept for muscle memory)"
	@echo "  make odds             — pull one odds snapshot now (set MLB_EV_LIVE_AFTER_ODDS=1 in .env to chain live_refresh)"
	@echo "  make docker-refresh   — one-shot full pipeline in the refresh Docker image"
	@echo ""
	@echo "  make live-refresh     — light loop: outcomes + schedule + predict + tracker + dashboard (cached model)"
	@echo "  make docker-live-up   — run Docker cron service for live_refresh (~every 2h)"
	@echo "  make docker-live-refresh — one-shot live_refresh inside Docker (same image as daemon)"
	@echo "  make odds-schedule-on/off/status — Mac awake: odds + live_refresh @ 9:00, 12:30, 18:00, 21:00 (see plist)"
	@echo "  make schedule-on/off/status — daily full refresh (07:00) via launchd"
	@echo "  make live-schedule-on/off/status — light refresh every 2h (awake)"
	@echo ""
	@echo "Override YEAR=YYYY to target a different season."

refresh:
	$(PYTHON) -m src.pipeline.daily_refresh --year $(YEAR)

refresh-data:
	$(PYTHON) -m src.pipeline.daily_refresh --year $(YEAR) --no-predict

project predict:
	$(PYTHON) -m src.pipeline.daily_refresh --year $(YEAR) \
	    --skip-outcomes --skip-boxscores --skip-statcast \
	    --skip-oaa --skip-lineups --skip-features

odds:
	$(PYTHON) -m src.ingest.fetch_odds

live-refresh:
	$(PYTHON) -m src.pipeline.live_refresh --year $(YEAR)

docker-live-up:
	docker compose up -d --build live

docker-live-logs:
	docker compose logs -f live

docker-live-refresh:
	docker compose run --rm live python -m src.pipeline.live_refresh --year $(YEAR)

docker-scheduler-up:
	docker compose up -d --build scheduler

docker-scheduler-logs:
	docker compose logs -f scheduler

docker-scheduler-down:
	docker compose stop scheduler


# ----- launchd scheduling (macOS native; replaces cron) -----------------------

schedule-on: infra/launchd.refresh.plist
	mkdir -p $(HOME)/Library/LaunchAgents
	sed "s|__REPO_ROOT__|$(PWD)|g; s|__PYTHON__|$(PWD)/$(PYTHON)|g" \
	    infra/launchd.refresh.plist > $(LAUNCHD_PLIST)
	launchctl unload $(LAUNCHD_PLIST) 2>/dev/null || true
	launchctl load $(LAUNCHD_PLIST)
	@echo "Installed launchd job: $(LAUNCHD_LABEL)"
	@echo "  Runs every day at 07:00 America/New_York."
	@echo "  Logs:  $(PWD)/data/predictions/launchd.{out,err}.log"

schedule-off:
	-launchctl unload $(LAUNCHD_PLIST) 2>/dev/null
	-rm -f $(LAUNCHD_PLIST)
	@echo "Removed launchd job."

schedule-status:
	@echo "plist file: $(LAUNCHD_PLIST)"
	@ls -la $(LAUNCHD_PLIST) 2>/dev/null || echo "  (not installed)"
	@echo ""
	@echo "launchctl list entry (if loaded):"
	@launchctl list | grep -F $(LAUNCHD_LABEL) || echo "  (not loaded)"
	@echo ""
	@echo "Last stdout/stderr (tail):"
	@tail -n 20 data/predictions/launchd.out.log 2>/dev/null || echo "  (no stdout log yet)"
	@tail -n 20 data/predictions/launchd.err.log 2>/dev/null || echo "  (no stderr log yet)"

LAUNCHD_LIVE_LABEL := com.dmatten.mlb-ev.live
LAUNCHD_LIVE_PLIST := $(HOME)/Library/LaunchAgents/$(LAUNCHD_LIVE_LABEL).plist

live-schedule-on: infra/launchd.live.plist
	mkdir -p $(HOME)/Library/LaunchAgents
	sed "s|__REPO_ROOT__|$(PWD)|g; s|__PYTHON__|$(PWD)/$(PYTHON)|g" \
	    infra/launchd.live.plist > $(LAUNCHD_LIVE_PLIST)
	launchctl unload $(LAUNCHD_LIVE_PLIST) 2>/dev/null || true
	launchctl load $(LAUNCHD_LIVE_PLIST)
	@echo "Installed launchd job: $(LAUNCHD_LIVE_LABEL) (every 2h while awake)"
	@echo "  Logs:  $(PWD)/data/predictions/launchd.live.{out,err}.log"

live-schedule-off:
	-launchctl unload $(LAUNCHD_LIVE_PLIST) 2>/dev/null
	-rm -f $(LAUNCHD_LIVE_PLIST)
	@echo "Removed launchd live job."

live-schedule-status:
	@echo "plist file: $(LAUNCHD_LIVE_PLIST)"
	@ls -la $(LAUNCHD_LIVE_PLIST) 2>/dev/null || echo "  (not installed)"
	@launchctl list | grep -F $(LAUNCHD_LIVE_LABEL) || echo "  (not loaded)"
	@echo "Last live stdout/stderr (tail):"
	@tail -n 15 data/predictions/launchd.live.out.log 2>/dev/null || echo "  (none)"
	@tail -n 15 data/predictions/launchd.live.err.log 2>/dev/null || echo "  (none)"

# ----- Odds + live_refresh at Docker scheduler times (laptop awake, macOS) -----

ODDS_LAUNCHD_LABEL := com.dmatten.mlb-ev.odds
ODDS_LAUNCHD_PLIST := $(HOME)/Library/LaunchAgents/$(ODDS_LAUNCHD_LABEL).plist

odds-schedule-on: infra/launchd.odds.plist
	mkdir -p $(HOME)/Library/LaunchAgents
	sed "s|__REPO_ROOT__|$(PWD)|g; s|__PYTHON__|$(PWD)/$(PYTHON)|g" \
	    infra/launchd.odds.plist > $(ODDS_LAUNCHD_PLIST)
	launchctl unload $(ODDS_LAUNCHD_PLIST) 2>/dev/null || true
	launchctl load $(ODDS_LAUNCHD_PLIST)
	@echo "Installed launchd job: $(ODDS_LAUNCHD_LABEL)"
	@echo "  fetch_odds --live-after at 8:00 / 11:30 / 17:00 / 20:00 local (set Mac TZ to Chicago to match Docker; Eastern Mac: edit plist to 9/12:30/18/21)."
	@echo "  Logs:  $(PWD)/data/predictions/launchd.odds.{out,err}.log"

odds-schedule-off:
	-launchctl unload $(ODDS_LAUNCHD_PLIST) 2>/dev/null
	-rm -f $(ODDS_LAUNCHD_PLIST)
	@echo "Removed odds launchd job."

odds-schedule-status:
	@echo "plist file: $(ODDS_LAUNCHD_PLIST)"
	@ls -la $(ODDS_LAUNCHD_PLIST) 2>/dev/null || echo "  (not installed)"
	@launchctl list | grep -F $(ODDS_LAUNCHD_LABEL) || echo "  (not loaded)"
	@echo "Last odds+live stdout/stderr (tail):"
	@tail -n 20 data/predictions/launchd.odds.out.log 2>/dev/null || echo "  (none)"
	@tail -n 20 data/predictions/launchd.odds.err.log 2>/dev/null || echo "  (none)"

