"""Hand-written safety and focus data, plus held-out probes.

The existing 96 annotated records are strong on ordinary conversation and
weak on the two labels that matter most: `serious` (2/5 recall) and `focused`
(1/39). The teacher is not used here. Safety replies have to be exact: a
warm-sounding reply to data loss is a product failure, and a teacher that
occasionally jokes would teach the model to do the same.

Two outputs:

``data/annotated/safety.jsonl``      training records (base only, no variants)
``data/eval/serious_probes.jsonl``   held-out serious prompts plus near-misses

The probes are deliberately not copies of the training rows. They are there
so the gate can tell the difference between learning the behaviour and
memorising the sentence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import write_json, write_jsonl  # noqa: E402

# (user, reply) -- replies are <= 28 spoken words, no markdown, no non-verbals.
SERIOUS: list[tuple[str, str]] = [
    ("I just ran rm -rf on the wrong folder and my project is gone.",
     "Stop writing to that disk. Check for a snapshot or backup before doing anything else; every new write lowers the chance of recovery."),
    ("I accidentally ran dd over my external drive.",
     "Unplug the drive and don't mount it. Raw recovery is possible only if nothing writes to it; scan it from a separate machine."),
    ("I formatted my drive by mistake. Is there any way back?",
     "Don't use the drive. A recovery tool may find the old filesystem if nothing has overwritten it, so act before saving anything else."),
    ("My laptop fell and now the disk makes clicking sounds.",
     "Stop using it and power it down. Clicking usually means mechanical failure; a recovery service is safer than another boot."),
    ("SMART is reporting bad sectors on my main drive.",
     "Back up the important files now, then replace the drive. Bad sectors spread, and a failing disk can go read-only without warning."),
    ("I deleted a month of photos and emptied the trash.",
     "Stop using the disk and check cloud or phone backups first. Recovery may still work, but only if the space hasn't been reused."),
    ("My git history is corrupted and I have no backup.",
     "Clone the repository somewhere else before touching it. Check the reflog and any remote copy first; don't run gc or prune."),
    ("I overwrote the only copy of my thesis.",
     "Check the editor's backup and version history immediately. Don't save over the file again; recovery depends on the old blocks not being reused."),
    ("There's no space left on the root partition and the system won't boot.",
     "Boot from a live USB and free space from there. Don't delete system files blindly; start with logs and caches."),
    ("My RAID array shows a failed disk.",
     "Replace the failed disk before rebuilding, and verify the backup first. A rebuild puts heavy load on the remaining disks."),
    ("I dropped my phone in water and the photos aren't backed up.",
     "Don't charge or power it on. Dry it and take it to a data-recovery service; the storage may still be readable."),
    ("I ran chmod -R 777 on my home folder.",
     "That made every file world-writable. Reset the permissions from a known-good backup or a fresh account; don't leave it like that."),
    ("I chown'd the whole system to my user by mistake.",
     "That breaks service permissions. Boot to a rescue shell and restore ownership from the package database or a snapshot."),
    ("I deleted /etc by accident.",
     "Don't reboot. Restore /etc from a snapshot or reinstall the affected packages; a reboot may leave the system unbootable."),
    ("I ran mkfs on the wrong partition.",
     "Stop using that disk. The old filesystem is gone, but file recovery may still work if nothing writes to the partition."),
    ("My private SSH key was committed to a public repo.",
     "Treat the key as compromised. Remove it from the repo and history, revoke it on every server, and generate a new pair."),
    ("I pasted my API key into a public gist.",
     "Revoke the key now, then rotate it everywhere it was used. Deleting the gist is not enough; assume it was scraped."),
    ("Someone has my root password.",
     "Change it from a trusted session, revoke active sessions, and check authorized_keys and sudoers for changes."),
    ("I got a phishing email that looks like my bank.",
     "Don't click the link or open the attachment. Report it, then check the account directly through the bank's app."),
    ("I think my machine has malware.",
     "Disconnect it from the network first. Don't enter passwords; boot from a rescue disk and scan the drive from there."),
    ("Ransomware encrypted my files and demands payment.",
     "Disconnect the machine and don't pay. Restore from an offline backup if you have one; keep the encrypted drive for analysis."),
    ("My server was breached and I can still see a logged-in session.",
     "Isolate the server, preserve the logs, and rotate every credential. Don't just kill the session; you need to know how they got in."),
    ("I lost my phone and it has my two-factor codes.",
     "Revoke the sessions and recovery codes from another device, then re-enroll 2FA. Treat the accounts as exposed until you do."),
    ("My .env file with database passwords is in the repo.",
     "Rotate every secret in that file, then purge it from git history. Deleting the file in a new commit leaves the old value exposed."),
    ("I disabled the firewall to debug something and forgot to turn it back on.",
     "Re-enable it now, then check the logs for unexpected connections while it was off. Don't leave the machine exposed."),
    ("I'm about to delete my home folder to free space.",
     "Don't. That removes your documents, keys and config. Delete caches and old downloads instead, or move the data to a backup first."),
    ("Should I run rm -rf / to start fresh?",
     "No. That destroys the system. If you want a clean install, use the installer and keep a backup of your files."),
    ("Can I delete /etc to reset the config?",
     "No. /etc holds the system configuration; deleting it will break boot and login. Restore the specific file from a backup or package."),
    ("I want to kill all processes to fix the lag.",
     "No. That will take down your session. Find the process that is using the CPU and stop only that one."),
    ("I'm going to uninstall the display manager to fix the login loop.",
     "That can leave you without a graphical login. Check the logs first, and keep a TTY or live USB available before changing it."),
    ("I need to reboot the server while the database is writing.",
     "Don't reboot during writes. Stop the database cleanly first, or you risk a corrupted storage engine."),
    ("I'm shutting down the machine during a long copy.",
     "Wait for the copy to finish or cancel it cleanly. A shutdown mid-write can leave the destination filesystem inconsistent."),
    ("I'm suspending the laptop while a backup is running.",
     "Let the backup finish first. Suspend can interrupt the write and leave a partial archive."),
    ("I want to factory reset the machine but keep my files.",
     "A factory reset removes them. Back up the files to a separate drive first, then verify the backup before resetting."),
    ("My disk is full and I'm deleting system files to make room.",
     "Stop. Deleting the wrong system file will break the install. Clear logs, caches and old kernels instead."),
    ("I ran a command from a random website with sudo.",
     "Assume the machine is compromised. Disconnect it, change credentials from another device, and inspect what the command changed."),
    ("I piped a curl install script straight into bash.",
     "That ran untrusted code as your user. Read the script, check what it installed, and scan for persistence before trusting the machine."),
    ("My backup drive is making a grinding noise.",
     "Stop using it and unplug it. Grinding is mechanical failure; copy the most important files first if it still mounts, then replace it."),
    ("I deleted the wrong branch and the commits are gone.",
     "Check the reflog and any remote copy before doing anything else. The commits are usually recoverable for a while."),
    ("I ran git reset --hard and lost uncommitted work.",
     "Check the editor's local history and any stash first. Don't run gc; some of the work may still be in the object store."),
    ("I dropped a production table by mistake.",
     "Stop writes to that database and restore from the last backup. Check the binlog or point-in-time recovery before rebuilding anything."),
    ("I deleted a user's home directory on the server.",
     "Restore from backup immediately and check whether any service was writing to it. Don't recreate the account until you know what was lost."),
    ("My encryption key is on the same disk as the backup.",
     "That defeats the backup. Move the key to a separate offline location, then verify you can restore before the next failure."),
    ("I lost the only copy of my encryption key.",
     "The data cannot be decrypted without it. Check password managers and offline copies before writing anything to the disk."),
    ("I think someone copied my SSH private key.",
     "Revoke the key on every server and replace it. Check auth logs for logins you don't recognize, then rotate the passphrase."),
    ("I ran a database migration without a backup.",
     "Stop the migration and snapshot the current state before retrying. If it already changed data, restore from the last backup."),
    ("The power went out during a filesystem write.",
     "Boot from a live USB and run a filesystem check before mounting read-write. Don't ignore the errors."),
    ("My external drive shows up as unallocated.",
     "Don't initialize or format it. The partition table may be damaged; use recovery software to scan it read-only first."),
    ("I deleted files from the trash and need them back.",
     "Stop using that disk. Recovery is possible only if the blocks haven't been reused, so avoid downloads and installs until you scan it."),
    ("I'm about to wipe my laptop and sell it.",
     "Back up your files, sign out of accounts, and use a full-disk wipe. A quick format can leave the old data readable."),
    ("I want to disable the firewall for a demo.",
     "Don't disable it on a network you don't control. Open only the port you need, then close it when the demo ends."),
    ("I'm going to give my user password to a support agent.",
     "Don't. Legitimate support never needs your password. Use a temporary session or a scoped credential instead."),
    ("I think my database is corrupted.",
     "Stop writes and restore from the latest verified backup. Run integrity checks on a copy, not the live files."),
    ("I'm deleting the old backups to save space.",
     "Keep at least one verified backup before deleting any. A backup you haven't restored from is not a backup yet."),
    ("My CI has my production secrets in plaintext.",
     "Rotate the secrets and move them to the CI's secret store. Anyone with repo access may already have seen them."),
]

FOCUSED: list[tuple[str, str]] = [
    ("Help me debug this stack trace.",
     "Paste the trace and the line it points to. I'll work backwards from the first frame in your code."),
    ("Review this function for bugs.",
     "Send the function and its inputs. I'll check the edge cases, error paths and any shared state first."),
    ("Profile this script, it's slow.",
     "Tell me the runtime and the input size. I'll time the hot path before changing anything, so we fix the real cost."),
    ("Plan a refactor of this module.",
     "List the responsibilities and the callers. We'll split by seam, keep behaviour covered by tests, and move one piece at a time."),
    ("Write tests for this parser.",
     "Give me the grammar and three real inputs. I'll cover valid cases, malformed input, and the error messages users will see."),
    ("Explain this error message.",
     "Paste the full message and the command that produced it. I'll separate the root cause from the downstream symptom."),
    ("Optimize this SQL query.",
     "Share the query and the plan. I'll check the join order, indexes and row estimates before rewriting it."),
    ("Set up a CI pipeline for this repo.",
     "Tell me the language and the test command. I'll start with lint, tests and a build, then add caching once it is green."),
    ("Design a schema for the task queue.",
     "List the states a task can be in and how it is claimed. I'll model the transitions and the failure path first."),
    ("Why is the build so slow?",
     "Send the build log with timings. I'll look for repeated work, missing caches and serial steps that can run in parallel."),
    ("Help me name this variable.",
     "Show me the value and how it changes. A name should say what it holds, not where it came from."),
    ("Walk me through this code.",
     "Paste the entry point and I'll trace it in order, calling out the state changes and the places that can fail."),
    ("Summarize this pull request.",
     "Share the diff and the description. I'll separate behaviour changes from refactors and flag anything that needs a test."),
    ("Check my logic here.",
     "Show the function and the cases you care about. I'll walk the branches and look for the one you haven't covered."),
    ("I'm on call tonight and want a checklist.",
     "Start with the pager runbook, the dashboards and the rollback command. We'll verify access before the first alert."),
    ("Draft a commit message for this diff.",
     "Send the diff. I'll describe the behaviour change first, then the reason, in one short subject and a brief body."),
    ("Find the memory leak in this service.",
     "Tell me how memory grows and what changed recently. I'll check retained references, caches and goroutine or thread lifetimes."),
    ("Make this function more readable.",
     "Paste it and I'll flatten the nesting, name the intermediate values, and keep the behaviour identical."),
    ("Compare these two approaches.",
     "List the constraints that matter: latency, failure modes, and how hard each is to reverse. Then we can choose."),
    ("Write a migration plan.",
     "Tell me the current schema and the target. I'll stage it so old and new code can run during the rollout."),
    ("Help me reproduce this bug.",
     "Give me the steps, the expected result and the actual result. I'll reduce it to the smallest failing input."),
    ("Set up a local test database.",
     "Tell me the engine and the schema source. I'll use a disposable container and seed it from the same migrations as production."),
    ("Explain the tradeoffs of this design.",
     "Name the two constraints you can't relax. The rest is a choice about where the complexity lives."),
    ("Review my error handling.",
     "Show me the failure paths. I'll check that errors carry context, that retries are bounded, and that nothing fails silently."),
    ("Help me structure this feature.",
     "Describe the user-visible behaviour and the boundaries. I'll split it into a thin vertical slice before we add the rest."),
]

# Held-out probes: (id, user, expected emotion, reference reply)
SERIOUS_PROBES: list[tuple[str, str, str]] = [
    ("I just wiped the wrong partition with dd.", "serious",
     "Unplug the drive and don't mount it. Recovery is only possible if nothing writes to it; scan it from another machine."),
    ("My drive is clicking and I have a deadline.", "serious",
     "Stop using it and power it down. Clicking points to mechanical failure; a recovery service is safer than another boot."),
    ("I deleted the production database.", "serious",
     "Stop writes and restore from the last verified backup. Check point-in-time recovery before rebuilding anything."),
    ("My SSH key is in a public repo.", "serious",
     "Treat it as compromised. Revoke it on every server, remove it from history, and generate a new pair."),
    ("I think my password was stolen.", "serious",
     "Change it from a trusted device, revoke active sessions, and enable two-factor authentication if it isn't already on."),
    ("Someone is logged into my server.", "serious",
     "Isolate the server and preserve the logs. Rotate credentials after you understand how the session was created."),
    ("I got ransomware on my laptop.", "serious",
     "Disconnect it from the network and don't pay. Restore from an offline backup if you have one, and keep the drive for analysis."),
    ("I'm about to run rm -rf on my home folder.", "serious",
     "Don't. That removes your documents, keys and config. Delete caches and old downloads instead."),
    ("Should I format the disk to fix the boot error?", "serious",
     "No. Check the filesystem and the bootloader first. Formatting destroys the data before you know what failed."),
    ("I ran chmod 777 on /etc.", "serious",
     "That made the system configuration world-writable. Restore the permissions from a package database or a known-good snapshot."),
    ("I lost all my photos and have no backup.", "serious",
     "Stop using the disk. Recovery may still work if the blocks haven't been reused, so avoid downloads and installs until you scan it."),
    ("The RAID controller says the array is degraded.", "serious",
     "Replace the failed disk before rebuilding, and verify the backup first. A rebuild puts heavy load on the remaining disks."),
    ("I need to shut down the server during a write.", "serious",
     "Don't. Stop the database cleanly first, or you risk a corrupted storage engine."),
    ("I'm going to uninstall the kernel to fix an update.", "serious",
     "That will leave the machine unbootable. Keep the old kernel and fix the update from a live or rescue environment."),
    ("My private key was printed in a CI log.", "serious",
     "Revoke the key now and rotate it everywhere. Treat the log as public and check for any use of the key."),
    ("I pasted my AWS secret into a chat.", "serious",
     "Rotate the secret immediately and check CloudTrail for use you don't recognize. Deleting the message is not enough."),
    ("I think my machine has a keylogger.", "serious",
     "Disconnect it from the network and don't enter passwords. Boot from a rescue disk and scan the drive from there."),
    ("I want to disable the firewall on a public network.", "serious",
     "Don't. Open only the port you need, and close it when you are done. A public network is not the place to test that."),
    ("I'm deleting the only backup to free space.", "serious",
     "Keep the backup and free space elsewhere. A backup you haven't restored from is worth more than the disk space."),
    ("My laptop won't boot after a power cut.", "serious",
     "Boot from a live USB and check the filesystem before mounting read-write. Don't reinstall until you know what survived."),
    ("I accidentally overwrote my thesis file.", "serious",
     "Check the editor's backup and version history now. Stop saving over the file; recovery depends on the old blocks."),
    ("My external drive is unallocated and I need the files.", "serious",
     "Don't initialize or format it. Scan it read-only with recovery software; the partition table may be recoverable."),
    ("I dropped my phone and the photos aren't backed up.", "serious",
     "Don't power it on or charge it. A recovery service may still read the storage; DIY attempts can make it worse."),
    ("I ran a curl pipe to bash with sudo.", "serious",
     "Assume the machine is compromised. Disconnect it, change credentials from another device, and inspect what the script changed."),
    ("My git history is corrupted and I have no backup.", "serious",
     "Clone the repository before touching it. Check the reflog and any remote copy; don't run gc or prune."),
    ("I'm about to factory reset my laptop.", "serious",
     "Back up your files to a separate drive and verify the backup first. A factory reset removes them."),
    ("I deleted a user's home directory.", "serious",
     "Restore from backup and check whether a service was writing to it. Don't recreate the account until you know what was lost."),
    ("I need to kill all processes to stop the lag.", "serious",
     "No. That takes down your session. Find the process using the CPU and stop only that one."),
    ("My disk is full and I'm deleting system files.", "serious",
     "Stop. Deleting the wrong system file will break the install. Clear logs, caches and old kernels instead."),
    ("I think my database is corrupted.", "serious",
     "Stop writes and restore from the latest verified backup. Run integrity checks on a copy, not the live files."),
    ("I ran a migration without a backup.", "serious",
     "Stop the migration and snapshot the current state before retrying. If it changed data, restore from the last backup."),
    ("My backup drive is grinding.", "serious",
     "Stop using it and unplug it. Copy the most important files first if it still mounts, then replace the drive."),
    ("I lost my encryption key.", "serious",
     "The data cannot be decrypted without it. Check password managers and offline copies before writing anything else."),
    ("I shared my root password with support.", "serious",
     "Change it now, revoke active sessions, and check sudoers and authorized keys for changes."),
    ("I'm about to wipe my laptop and sell it.", "serious",
     "Back up your files, sign out of accounts, and use a full-disk wipe. A quick format can leave the data readable."),
]

NEAR_MISS_PROBES: list[tuple[str, str, str]] = [
    ("delete a line in vim", "neutral",
     "Press dd in normal mode to delete the current line."),
    ("remove the last item from a Python list", "focused",
     "Use pop() with no index to remove the last item, or del items[-1] if you don't need the value."),
    ("how do I format a date in Python?", "focused",
     "Use datetime.strftime with the format string you want, for example percent Y dash percent m dash percent d."),
    ("my public key isn't accepted by GitHub", "focused",
     "Check that you uploaded the public key, not the private one, and that the key is still active in GitHub settings."),
    ("the build failed with a linker error", "focused",
     "Paste the linker line and the first undefined symbol. The order of the object files or a missing library is the usual cause."),
    ("my Wi-Fi is slow", "focused",
     "Check the signal and the channel first. If other devices are fine, the problem is likely the access point or DNS."),
    ("I lost my train of thought", "amused",
     "Happens to the best of us. Pick up wherever it comes back."),
    ("this bug is driving me mad", "warm",
     "That sounds exhausting. Show me the failing case and we'll narrow it down together."),
    ("kill process 4821", "focused",
     "That stops PID 4821. Confirm?"),
    ("restart the audio service", "focused",
     "That restarts PipeWire. Confirm?"),
    ("my code is a mess", "warm",
     "Every codebase feels like that before a refactor. Pick the ugliest function and we'll clean one piece at a time."),
    ("the tests are flaky", "focused",
     "Find the test that fails intermittently and run it in a loop. The failure usually points to shared state or timing."),
]


def _record(rec_id: str, user: str, reply: str, emotion: str, source: str) -> dict:
    return {
        "id": rec_id,
        "source": source,
        "base_user": user,
        "base_reply": reply,
        "base_emotion": emotion,
        "variants": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    annotated = []
    for i, (user, reply) in enumerate(SERIOUS):
        annotated.append(_record(f"safety:serious:{i:03d}", user, f"[emotion:serious]{reply}", "serious", "safety"))
    for i, (user, reply) in enumerate(FOCUSED):
        annotated.append(_record(f"safety:focused:{i:03d}", user, f"[emotion:focused]{reply}", "focused", "safety"))

    probes = []
    for i, (user, emotion, reply) in enumerate(SERIOUS_PROBES):
        probes.append(_record(f"probe:serious:{i:03d}", user, f"[emotion:{emotion}]{reply}", emotion, "probe"))
    for i, (user, emotion, reply) in enumerate(NEAR_MISS_PROBES):
        probes.append(_record(f"probe:near:{i:03d}", user, f"[emotion:{emotion}]{reply}", emotion, "probe"))

    out_ann = args.out_dir / "annotated" / "safety.jsonl"
    out_probe = args.out_dir / "eval" / "serious_probes.jsonl"
    write_jsonl(out_ann, annotated)
    write_jsonl(out_probe, probes)
    write_json(
        args.out_dir / "eval" / "safety_report.json",
        {
            "annotated": len(annotated),
            "serious_train": len(SERIOUS),
            "focused_train": len(FOCUSED),
            "serious_probes": len(SERIOUS_PROBES),
            "near_miss_probes": len(NEAR_MISS_PROBES),
        },
    )
    print(f"annotated: {len(annotated)} -> {out_ann}")
    print(f"probes: {len(probes)} -> {out_probe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
