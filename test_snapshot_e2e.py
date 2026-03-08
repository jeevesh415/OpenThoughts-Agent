#!/usr/bin/env python3
"""E2E test: pre-build a snapshot for RL region and check its properties.

Tests whether Harbor's runtime snapshot.get() can find a snapshot that was
pre-built using the us region with region_id='RL'.
"""

import os
import sys

# Known snapshot from the failing job
SNAPSHOT_NAME = "harbor__a692f14ca7fe__RL__snapshot"

def main():
    from daytona import Daytona, DaytonaConfig
    from daytona.common.errors import DaytonaNotFoundError

    # =========================================================================
    # Step 1: Check snapshot from the "us" client (how pre-build sees it)
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Query snapshot from 'us'-targeted client (pre-build)")
    print("=" * 60)
    us_client = Daytona(DaytonaConfig(target="us"))

    try:
        snap = us_client.snapshot.get(SNAPSHOT_NAME)
        print(f"  Found: {snap.name}")
        print(f"  state:      {snap.state}")
        print(f"  image_name: {getattr(snap, 'image_name', 'N/A')}")
        print(f"  region_ids: {getattr(snap, 'region_ids', 'N/A')}")
        print(f"  id:         {getattr(snap, 'id', 'N/A')}")
        # Dump all attributes
        print(f"  All attrs:  {[a for a in dir(snap) if not a.startswith('_')]}")
    except DaytonaNotFoundError:
        print(f"  NOT FOUND from us client")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # =========================================================================
    # Step 2: Check snapshot from the "RL" client (how runtime sees it)
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 2: Query snapshot from 'RL'-targeted client (runtime)")
    print("=" * 60)
    rl_client = Daytona(DaytonaConfig(target="RL"))

    try:
        snap = rl_client.snapshot.get(SNAPSHOT_NAME)
        print(f"  Found: {snap.name}")
        print(f"  state:      {snap.state}")
        print(f"  image_name: {getattr(snap, 'image_name', 'N/A')}")
        print(f"  region_ids: {getattr(snap, 'region_ids', 'N/A')}")
        print(f"  id:         {getattr(snap, 'id', 'N/A')}")
    except DaytonaNotFoundError:
        print(f"  NOT FOUND from RL client")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

    # =========================================================================
    # Step 3: Check snapshot from a no-target client (default)
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 3: Query snapshot from default client (no target)")
    print("=" * 60)

    # Temporarily unset DAYTONA_TARGET to get a default client
    old_target = os.environ.pop("DAYTONA_TARGET", None)
    try:
        default_client = Daytona(DaytonaConfig())
        try:
            snap = default_client.snapshot.get(SNAPSHOT_NAME)
            print(f"  Found: {snap.name}")
            print(f"  state:      {snap.state}")
            print(f"  region_ids: {getattr(snap, 'region_ids', 'N/A')}")
        except DaytonaNotFoundError:
            print(f"  NOT FOUND from default client")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    finally:
        if old_target is not None:
            os.environ["DAYTONA_TARGET"] = old_target

    # =========================================================================
    # Step 4: List snapshots and check region_ids on the list items
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 4: List all snapshots (from us client), check regions")
    print("=" * 60)
    snap_list = us_client.snapshot.list(limit=50)
    print(f"  Total snapshots in org: {snap_list.total}")
    for item in snap_list.items:
        name = getattr(item, 'name', '?')
        state = getattr(item, 'state', '?')
        regions = getattr(item, 'region_ids', None)
        if "RL" in name or (regions and "RL" in str(regions)):
            print(f"  {name}: state={state}, region_ids={regions}")

    # =========================================================================
    # Step 5: Simulate Harbor runtime - set DAYTONA_TARGET=RL, create client
    #         with no config (like AsyncDaytona() in Harbor), try get()
    # =========================================================================
    print()
    print("=" * 60)
    print("STEP 5: Simulate Harbor runtime (DAYTONA_TARGET=RL, bare client)")
    print("=" * 60)
    os.environ["DAYTONA_TARGET"] = "RL"
    try:
        harbor_client = Daytona()  # No config, reads from env like Harbor does
        print(f"  Client target: {harbor_client._target}")
        try:
            snap = harbor_client.snapshot.get(SNAPSHOT_NAME)
            print(f"  snapshot.get() succeeded:")
            print(f"    name:       {snap.name}")
            print(f"    state:      {snap.state}")
            print(f"    region_ids: {getattr(snap, 'region_ids', 'N/A')}")
        except DaytonaNotFoundError:
            print(f"  snapshot.get() raised DaytonaNotFoundError")
        except Exception as e:
            print(f"  snapshot.get() raised {type(e).__name__}: {e}")
    finally:
        os.environ.pop("DAYTONA_TARGET", None)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
