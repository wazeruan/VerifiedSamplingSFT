#!/usr/bin/env bash
# Drive the cluster from your workstation over SSH (uses your ~/.ssh/config host alias).
#
#   slurm/remote.sh sync    --host HOST --remote-dir DIR
#   slurm/remote.sh submit  --host HOST --remote-dir DIR -- STAGE --account ACCT [submit.sh opts]
#   slurm/remote.sh status  --host HOST --remote-dir DIR RUN_DIR
#   slurm/remote.sh results --host HOST --remote-dir DIR RUN_DIR
#   slurm/remote.sh tail    --host HOST --remote-dir DIR LOG_PATH [N]
#
# `sync` copies the code trees (src, slurm, configs, tests) and project files to DIR;
# --delete applies only inside those trees.
# It only deletes inside a directory it created itself (marked by .vsft-checkout); a
# non-empty directory without that marker is refused. DIR may start with ~/ (remote home).
# Queued jobs are unaffected by later syncs: each run executes its own code snapshot.
# `submit` syncs first, then runs slurm/submit.sh on the login node; its exit code
# (0 STARTED, 10 PENDING, 1 FAILED, 3 UNKNOWN, 4 REJECTED, 2 usage) is passed through.
# Credentials are never passed on the command line: authentication is your SSH setup.
set -Eeuo pipefail

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 2; }

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
CMD="$1"; shift
case "$CMD" in sync|submit|status|results|tail) ;; -h|--help) usage; exit 0 ;; *) die "unknown command '$CMD'" ;; esac

HOST="" REMOTE_DIR="" POS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) [[ -n "${2:-}" ]] || die "--host requires a value"; HOST="$2"; shift 2 ;;
    --remote-dir) [[ -n "${2:-}" ]] || die "--remote-dir requires a value"; REMOTE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; POS+=("$@"); break ;;
    -*) die "unknown option '$1'" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[[ -n "$HOST" ]] || die "--host is required"
[[ -n "$REMOTE_DIR" ]] || die "--remote-dir is required"
[[ "$REMOTE_DIR" != *"'"* && "$REMOTE_DIR" != *[[:space:]]* ]] || die "--remote-dir must not contain quotes or whitespace (rsync remote paths are split on spaces by older clients)"
# shellcheck disable=SC2088  # literal "~" patterns are intended
case "$REMOTE_DIR" in
  /|"~"|"~/"|/home|/home/|/project|/project/|/scratch|/scratch/) die "refusing a top-level --remote-dir: $REMOTE_DIR" ;;
esac
command -v ssh >/dev/null || die "ssh not found"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
q() { printf '%q ' "$@"; }
# Remote-side expression for the directory: ~/x expands to the *remote* home.
# shellcheck disable=SC2088  # a literal "~/" prefix is intended (expanded remotely)
if [[ "$REMOTE_DIR" == "~/"* ]]; then
  RDIR_EXPR="\"\$HOME\"/$(printf '%q' "${REMOTE_DIR#\~/}")"
  RSYNC_DEST="$HOST:${REMOTE_DIR}"
else
  RDIR_EXPR="$(printf '%q' "$REMOTE_DIR")"
  RSYNC_DEST="$HOST:$REMOTE_DIR"
fi
remote() { ssh -o BatchMode=yes "$HOST" "cd $RDIR_EXPR && $*"; }

do_sync() {
  command -v rsync >/dev/null || die "rsync not found"
  # Safety: only sync (with --delete) into an empty dir or one marked as our checkout.
  ssh -o BatchMode=yes "$HOST" "d=$RDIR_EXPR; [ \"\$d\" != \"\$HOME\" ] || exit 9;
    if [ -e \"\$d\" ] && [ ! -f \"\$d/.vsft-checkout\" ] && [ -n \"\$(ls -A \"\$d\" 2>/dev/null)\" ]; then exit 8; fi;
    mkdir -p \"\$d\" && touch \"\$d/.vsft-checkout\"" || {
    rc=$?
    [[ "$rc" -eq 8 ]] && die "$HOST:$REMOTE_DIR is a non-empty directory that is not a vsft checkout; refusing to rsync --delete into it"
    [[ "$rc" -eq 9 ]] && die "--remote-dir resolves to the remote home directory; refusing"
    die "ssh to $HOST failed (exit $rc)"
  }
  # Deletion is confined to the tracked code trees; anything else in DIR (run dirs, venvs,
  # caches the user put there) is never touched. Top-level files are copied, not pruned.
  local tree
  for tree in src slurm configs tests; do
    [[ -d "$LOCAL_DIR/$tree" ]] || continue
    rsync -az --delete-after --exclude '__pycache__/' --exclude '.pytest_cache/' \
      "$LOCAL_DIR/$tree/" "$RSYNC_DEST/$tree/"
  done
  local files=()
  for f in pyproject.toml uv.lock README.md .python-version .gitignore; do
    [[ -f "$LOCAL_DIR/$f" ]] && files+=("$LOCAL_DIR/$f")
  done
  rsync -az "${files[@]}" "$RSYNC_DEST/"
  echo "synced $LOCAL_DIR -> $HOST:$REMOTE_DIR"
}

case "$CMD" in
  sync) do_sync ;;
  submit)
    [[ ${#POS[@]} -ge 1 ]] || die "submit needs: -- STAGE --account ACCT [...]"
    do_sync
    rc=0; remote "slurm/submit.sh $(q "${POS[@]}")" || rc=$?
    exit "$rc" ;;
  status|results)
    [[ ${#POS[@]} -eq 1 ]] || die "$CMD needs RUN_DIR"
    remote "slurm/$CMD.sh $(q "${POS[0]}")" ;;
  tail)
    [[ ${#POS[@]} -ge 1 ]] || die "tail needs LOG_PATH"
    n="${POS[1]:-200}"; [[ "$n" =~ ^[0-9]+$ ]] || die "N must be an integer"
    remote "tail -n $n $(q "${POS[0]}")" ;;
esac
