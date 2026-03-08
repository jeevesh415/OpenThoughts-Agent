#!/usr/bin/env python3
"""Test: can we create a sandbox from the pre-built RL snapshot?

This simulates what Harbor does at runtime when DAYTONA_TARGET=RL.
"""

import os

SNAPSHOT_NAME = "harbor__a692f14ca7fe__RL__snapshot"


def main():
    from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams

    # Simulate Harbor runtime: DAYTONA_TARGET=RL, bare client
    os.environ["DAYTONA_TARGET"] = "RL"
    try:
        client = Daytona()
        print(f"Client target: {client._target}")

        print(f"\nAttempting to create sandbox from snapshot: {SNAPSHOT_NAME}")
        print("(This is exactly what Harbor's start() does)")

        params = CreateSandboxFromSnapshotParams(
            snapshot=SNAPSHOT_NAME,
            auto_stop_interval=0,
            auto_delete_interval=5,  # Auto-delete after 5 min
            ephemeral=True,
        )

        try:
            sandbox = client.create(params=params, timeout=120)
            print(f"  SUCCESS! Sandbox created: {sandbox.id}")
            print(f"  Cleaning up...")
            try:
                client.delete(sandbox, force=True)
                print(f"  Sandbox deleted.")
            except Exception as e:
                print(f"  WARNING: cleanup failed: {e}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print(f"\n  This is the error Harbor sees at runtime!")

        # Now try with us-targeted client but still creating in RL context
        print(f"\n{'='*60}")
        print("Retry: create sandbox from 'us' client with same snapshot")
        print("=" * 60)
        us_client = Daytona(DaytonaConfig(target="us"))
        try:
            sandbox = us_client.create(params=params, timeout=120)
            print(f"  SUCCESS! Sandbox created: {sandbox.id}")
            print(f"  Cleaning up...")
            try:
                us_client.delete(sandbox, force=True)
                print(f"  Sandbox deleted.")
            except Exception as e:
                print(f"  WARNING: cleanup failed: {e}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")

    finally:
        os.environ.pop("DAYTONA_TARGET", None)


if __name__ == "__main__":
    main()
