#!/usr/bin/env python3
"""Send a notification mail from the cluster, without SLURM and without an MTA.

    notify.py --to you@example.com --subject "..." [--body "..."]     (or body on stdin)

WHY this exists: SLURM's `--mail-type` is accepted here and silently discarded.
`scontrol show config` reports `MailProg = /bin/mail`, and that binary is not
installed -- nor is sendmail, mailx or postfix. Jobs mail nobody.

What does work is talking to the campus relay directly. `smtp.tau.ac.il`
(post.tau.ac.il, 132.66.3.150) answers on port 25 from the login and compute
nodes, needs no authentication, and accepts external recipients -- verified by
`RCPT TO:<...@gmail.com>` returning `250 2.1.5 Ok`. Note `galbarak2@tau.ac.il`
is not a deliverable local address ("User unknown in local recipient table"), so
it works as an envelope sender but do not expect replies to reach anyone.

Exits non-zero on failure so a caller can log it, but a notification that cannot
be sent must never take down the job it was reporting on -- callers should
ignore the exit status rather than abort.
"""
from __future__ import annotations

import argparse
import datetime
import smtplib
import socket
import sys
from email.message import EmailMessage

RELAY = "smtp.tau.ac.il"
PORT = 25


def send(to: str, subject: str, body: str, sender: str | None = None, timeout: int = 30) -> None:
    host = socket.getfqdn()
    msg = EmailMessage()
    msg["From"] = sender or f"{__import__('getpass').getuser()}@tau.ac.il"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(
        f"{body.rstrip()}\n\n--\n"
        f"sent {datetime.datetime.now():%Y-%m-%d %H:%M:%S} from {host}\n"
        f"by Untaught's notify.py (SLURM --mail-type does not work on this cluster)\n")
    with smtplib.SMTP(RELAY, PORT, timeout=timeout) as s:
        s.ehlo(host)
        refused = s.send_message(msg)
    if refused:
        raise RuntimeError(f"relay refused: {refused}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--to", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", default=None, help="default: read stdin")
    ap.add_argument("--from", dest="sender", default=None)
    args = ap.parse_args()
    body = args.body if args.body is not None else sys.stdin.read()
    try:
        send(args.to, args.subject, body, args.sender)
    except Exception as e:                       # noqa: BLE001 - report, never raise
        print(f"notify: FAILED to send: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"notify: sent to {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
