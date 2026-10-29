# sample_competitors/ — MOCK data

> ⚠️ **Everything in this folder is invented for a portfolio demo.** Sumzilla, Ganitron Live and
> NumeroNest AI are **fictional companies**. Their prices, plans and changelogs are made up, and their
> URLs use the reserved `.example` domain so they can never point at a real business.
> The `edumaths360/profile.yaml` file is an **illustrative** product profile for the "us" side of the battlecard.

| Folder | What it simulates |
|---|---|
| `edumaths360/profile.yaml` | Our own product profile: plans, capabilities, differentiators, counter-positioning |
| `sumzilla/` | Budget, gamified practice app. Current pricing page, a June 2026 pricing snapshot (baseline) and a changelog |
| `ganitron/` | Premium live-class tutoring. Current pricing page, June 2026 snapshot and changelog |
| `numeronest/` | Freemium AI tutor. Current pricing page, June 2026 snapshot and changelog |

The `*_2026-06.html` files play the role of "what the page looked like last time we checked", so the agent
can show real pricing shifts on the very first run. In live mode the agent uses its own cache for this instead.
