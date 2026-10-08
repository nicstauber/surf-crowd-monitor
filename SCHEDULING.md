# Scheduling & alerts

GitHub's built-in `schedule:` trigger is best-effort. From mid-September to
early October 2026 it delivered only **4–7 of the ~72** runs requested each
day, and many of those arrived hours late. Manual (`workflow_dispatch`) runs
start within seconds, so an outside timer starts the workflow instead.

```
 cron-job.org ──every 15 min──▶ GitHub "run workflow" API ──▶ Surf Sample job
                                                                │
                                            success → ping ─────┤
                                            failure → ping /fail┘
                                                    ▼
                                             healthchecks.io ──no ping 1h──▶ email you
```

Both services are free. Setup takes about 15 minutes.

---

## Part 1: the timer (cron-job.org)

### 1. Create a GitHub token that can only start workflows

1. Go to **GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token**.
2. **Name:** `surf-sample-trigger`
3. **Expiration:** your choice (1 year max). Put a reminder in your calendar.
4. **Repository access:** *Only select repositories* → `surf-crowd-monitor`
5. **Permissions → Repository → Actions:** **Read and write**. Leave everything else as *No access*.
6. Click **Generate** and copy the token (starts with `github_pat_`).

### 2. Create the cron job

1. Sign up at **https://cron-job.org** and click **Create cronjob**.
2. **Title:** `Surf Sample`
3. **URL:**
   ```
   https://api.github.com/repos/nicstauber/surf-crowd-monitor/actions/workflows/sample.yml/dispatches
   ```
4. **Schedule:** *Every 15 minutes*. Night runs are fine: the workflow sees it's
   dark and exits in about 2 seconds, and Actions minutes are free on a public repo.
5. Open the **Advanced** tab:
   - **Request method:** `POST`
   - **Headers** (add three):

     | Key | Value |
     |---|---|
     | `Accept` | `application/vnd.github+json` |
     | `Authorization` | `Bearer github_pat_…` (your token) |
     | `X-GitHub-Api-Version` | `2022-11-28` |

   - **Request body:**
     ```json
     {"ref":"main"}
     ```
6. Click **Test run**. A good result is **HTTP 204** (no content).
   Then check the repo's **Actions** tab; a new *Surf Sample* run should appear.
7. **Save.**

| If the test returns | It means |
|---|---|
| `204` | ✅ Working |
| `401` | Token is wrong or expired |
| `403` / `404` | Token is missing **Actions: Read and write**, or isn't scoped to this repo |
| `422` | Body is wrong; it must be exactly `{"ref":"main"}` |

### 3. After a day or two of good runs: remove GitHub's own schedule

Delete the `schedule:` block (the `- cron:` line and the comment above it) from
`.github/workflows/sample.yml`, and keep `workflow_dispatch:`. That stops late
GitHub ticks from adding extra samples on top of the cron-job.org ones.

---

## Part 2: the alert (healthchecks.io)

1. Sign up at **https://healthchecks.io** and click **Add Check**.
2. **Name:** `Surf Sample`
3. **Schedule** → choose **Cron**:
   - **Cron expression:** `*/15 7-16 * * *`
   - **Time zone:** `America/Los_Angeles`
   - **Grace time:** `45 minutes`

   7am–5pm Pacific is inside the sampling window all year, so no alerts fire
   at night or on winter evenings.
4. Copy the check's **ping URL** (looks like `https://hc-ping.com/<uuid>`).
5. In GitHub go to **repo → Settings → Secrets and variables → Actions → New repository secret**:
   - **Name:** `HEALTHCHECK_URL`
   - **Value:** the ping URL
6. Under **Integrations**, make sure email is on (or add SMS or Slack).

### What you'll get

| Situation | Alert |
|---|---|
| A sample fails (DB write error, every cam down) | Right away, via `/fail` ping |
| Runs stop arriving (timer broken, token expired) | ~1 hour after the last good run |
| One cam goes dark but others are fine | Not from healthchecks. Run `python3 scripts/verify_stack.py`; its *every spot reporting* check names the spot. |
