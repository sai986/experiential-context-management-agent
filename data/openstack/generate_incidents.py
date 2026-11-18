"""
Generate OpenStack multi-service incidents for the experiential context-management pipeline.

Reads data/openstack/openstack_structured.csv (loghub OpenStack_2k) and produces:
  - data/openstack/incidents.jsonl      — master list (180 incidents)
  - data/openstack/train.jsonl          — 125 incidents (Phase 1 training)
  - data/openstack/validation.jsonl     —  28 incidents
  - data/openstack/test.jsonl           —  27 incidents (Phase 2/3 evaluation)

The 180 incidents are 30 per family × 6 families, interleaved.  Each incident
has 13-15 observations in raw_text: 5-7 real Nova log lines from the CSV window
plus 8 background noise lines from multiple services (Nova, Neutron, Keystone,
Cinder, Glance), giving a realistic multi-service triage context.

Families
--------
  hard_constraint     — Nova VM migration RAM constraint
  delayed_dependency  — Neutron security group sync gate
  dependency_chain    — 4-step Keystone -> Nova -> Neutron -> Cinder launch sequence
  false_distractor    — Glance image error mis-labelled as Nova scheduler failure
  negation_critical   — Keystone token renewal auto-renew prohibition
  synonym_match       — Nova API rate-limit billing report

Usage
-----
    python3 data/openstack/generate_incidents.py

Output
------
    data/openstack/incidents.jsonl     (180 lines — master list)
    data/openstack/train.jsonl         (125 lines)
    data/openstack/validation.jsonl    ( 28 lines)
    data/openstack/test.jsonl          ( 27 lines)
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import uuid
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_HERE, "openstack_structured.csv")
OUT_PATH = os.path.join(_HERE, "incidents.jsonl")
TRAIN_PATH = os.path.join(_HERE, "train.jsonl")
VAL_PATH   = os.path.join(_HERE, "validation.jsonl")
TEST_PATH  = os.path.join(_HERE, "test.jsonl")

N_PER_FAMILY = 30   # 30 × 6 = 180 total incidents
TRAIN_N      = 125
VAL_N        = 28
TEST_N       = 27   # 125 + 28 + 27 = 180

# ---------------------------------------------------------------------------
# Load the CSV
# ---------------------------------------------------------------------------
def _load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Realistic UUID / ID generators
# ---------------------------------------------------------------------------
def _reqid() -> str:
    return "req-" + str(uuid.UUID(int=random.getrandbits(128)))

def _instid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128)))

def _volid() -> str:
    return f"vol-{uuid.uuid4().hex[:8]}"

def _imgid() -> str:
    return f"img-{uuid.uuid4().hex[:8]}"

def _portid() -> str:
    return f"port-{uuid.uuid4().hex[:8]}"

def _ip() -> str:
    return f"10.{random.randint(0,15)}.{random.randint(0,255)}.{random.randint(1,254)}"

def _host() -> str:
    return f"compute-{random.randint(1,8)}"

def _user() -> str:
    return f"user-{uuid.uuid4().hex[:8]}"

def _proj() -> str:
    return f"proj-{uuid.uuid4().hex[:8]}"

def _ms() -> str:
    return f"{random.randint(50,999)}"


# ---------------------------------------------------------------------------
# Multi-service log line generators
# ---------------------------------------------------------------------------
NOVA_TEMPLATES = [
    "[nova.compute.manager] [INFO] [instance: {inst}] VM Started (Lifecycle Event)",
    "[nova.compute.manager] [INFO] [instance: {inst}] Claim successful on {host}",
    "[nova.osapi_compute.wsgi.server] [INFO] {ip} GET /v2/{proj}/servers/detail HTTP/1.1 status: 200",
    "[nova.compute.resource_tracker] [INFO] Final resource view host={host} free_ram_mb=95746",
    "[nova.virt.libvirt.imagecache] [WARNING] Unknown base file: /var/lib/nova/instances/_base/{img}",
    "[nova.compute.manager] [INFO] [instance: {inst}] During sync_power_state the instance has a pending task (spawning)",
    "[nova.scheduler.manager] [DEBUG] Selected host {host} for instance {inst}",
    "[nova.api.openstack.wsgi] [INFO] request completed in 0.{ms}s req_id={req}",
]

NEUTRON_TEMPLATES = [
    "[neutron.wsgi.server] [INFO] {ip} POST /v2.0/ports HTTP/1.1 status: 201 req_id={req}",
    "[neutron.plugins.ml2.managers] [INFO] Attempting to bind port {port} on host {host}",
    "[neutron.agent.linux.iptables_manager] [INFO] Refresh security group rules for instance {inst}",
    "[neutron.wsgi.server] [WARNING] security group rule sync pending for port {port} req_id={req}",
    "[neutron.agent.securitygroups_rpc] [INFO] Security group member updated [{inst}]",
    "[neutron.wsgi.server] [DEBUG] GET /v2.0/ports 200 latency=0.{ms}ms req_id={req}",
]

KEYSTONE_TEMPLATES = [
    "[keystone.middleware.auth_token] [INFO] Starting Keystone authentication for user {user}",
    "[keystone.token.provider] [INFO] Issuing token for user {user} project {proj}",
    "[keystone.common.wsgi] [INFO] GET /v3/auth/tokens HTTP/1.1 200 req_id={req}",
    "[keystone.token.provider] [WARNING] Token validation failed for user {user}: expired req_id={req}",
    "[keystone.middleware.auth_token] [INFO] token validated for user admin project={proj}",
]

CINDER_TEMPLATES = [
    "[cinder.api.v2.volumes] [INFO] Creating volume size=20 for instance {inst} req_id={req}",
    "[cinder.volume.manager] [INFO] Attaching volume {vol} to instance {inst}",
    "[cinder.scheduler.manager] [INFO] Scheduling volume {vol} to host {host}",
    "[cinder.volume.drivers.lvm] [INFO] Volume {vol} attached at /dev/vdb",
    "[cinder.api.v2] [DEBUG] volume status check OK {vol} req_id={req}",
]

GLANCE_TEMPLATES = [
    "[glance.api.v2.images] [INFO] GET /v2/images/{img} HTTP/1.1 200 req_id={req}",
    "[glance.store.http] [INFO] Downloading image {img} from http://glance-store/{img}",
    "[glance.api.v2.images] [ERROR] Image {img} download failed: storage backend timeout req_id={req}",
    "[glance.registry.client] [INFO] Image {img} registered for instance {inst}",
    "[glance.store.http] [WARNING] Retrying image {img} download (attempt 2) req_id={req}",
    "[glance.registry] [DEBUG] image cache hit for {img}",
]


def _fill(template: str, **kwargs) -> str:
    inst = kwargs.get("inst", _instid())
    host = kwargs.get("host", _host())
    req  = kwargs.get("req",  _reqid())
    vol  = kwargs.get("vol",  _volid())
    img  = kwargs.get("img",  _imgid())
    port = kwargs.get("port", _portid())
    ip   = kwargs.get("ip",   _ip())
    user = kwargs.get("user", _user())
    proj = kwargs.get("proj", _proj())
    ms   = kwargs.get("ms",   _ms())
    return template.format(
        inst=inst, host=host, req=req, vol=vol, img=img,
        port=port, ip=ip, user=user, proj=proj, ms=ms,
    )


def _noise_lines(n: int = 8, inst: str = "", req: str = "") -> list[str]:
    """Generate n background noise lines from multiple services."""
    inst = inst or _instid()
    req  = req  or _reqid()
    all_templates = (
        NOVA_TEMPLATES + NEUTRON_TEMPLATES + KEYSTONE_TEMPLATES
        + CINDER_TEMPLATES + GLANCE_TEMPLATES
    )
    rng = random.Random(hash(inst) % (2**32))
    chosen = rng.sample(all_templates, min(n, len(all_templates)))
    return [_fill(t, inst=inst, req=req) for t in chosen]


# ---------------------------------------------------------------------------
# Nova window sampler
# ---------------------------------------------------------------------------
def _nova_window(rows: list[dict], center: int, before: int = 3, after: int = 6) -> list[str]:
    start = max(0, center - before)
    end   = min(len(rows), center + after + 1)
    lines = []
    for r in rows[start:end]:
        comp    = r["Component"]
        lvl     = r["Level"]
        content = r["Content"].replace('"', "'").strip()[:120]
        lines.append(f"[{comp}] [{lvl}] {content}")
    return lines


def _extract_instance_uuid(nova_lines: list[str]) -> str:
    """Try to find an instance UUID from the Nova log lines."""
    uuid_re = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    for line in nova_lines:
        m = uuid_re.search(line)
        if m:
            return m.group(0)
    return _instid()


def _pick_centers(rows: list[dict], n: int, seed_offset: int = 0) -> list[int]:
    """Pick n evenly-spread, deduped-by-EventTemplate centers."""
    seen: set[str] = set()
    candidates = []
    step = max(1, len(rows) // (n * 2))
    for idx in range(0, len(rows), step):
        tmpl = rows[idx].get("EventTemplate", str(idx))
        if tmpl not in seen:
            seen.add(tmpl)
            candidates.append(idx)
    rng = random.Random(42 + seed_offset)
    rng.shuffle(candidates)
    # If not enough unique templates, fill with spaced indices
    if len(candidates) < n:
        extra = [i for i in range(0, len(rows), max(1, len(rows) // n))
                 if i not in candidates]
        candidates += extra
    return candidates[:n]


# ---------------------------------------------------------------------------
# Family-specific incident builders
# ---------------------------------------------------------------------------

def _build_hard_constraint(
    idx: int, nova_lines: list[str], inst: str
) -> dict:
    req = _reqid()
    neutron_lines = [
        _fill(NEUTRON_TEMPLATES[0], inst=inst, req=req),
        _fill(NOVA_TEMPLATES[6], inst=inst),
    ]
    noise = _noise_lines(8, inst=inst, req=req)
    # Interleave: nova[0:3] + neutron + noise[:4] + nova[3:] + noise[4:]
    mid = max(2, len(nova_lines) // 2)
    all_lines = nova_lines[:mid] + neutron_lines + noise[:4] + nova_lines[mid:] + noise[4:]
    all_lines = all_lines[:15]

    constraint = (
        f"CONSTRAINT: Nova compute [instance: {inst}] VM migrations must NOT use "
        f"live migration when host has less than 2GB free RAM. "
        f"Always perform a cold migration: stop the VM first, then migrate."
    )
    raw_text = constraint + "\n" + "\n".join(all_lines)

    return {
        "family":          "hard_constraint",
        "failure_mode":    "constraint_compression",
        "services":        ["nova", "neutron"],
        "question":        "What constraint must be respected when migrating this VM?",
        "required_outputs": ["cold migration", "stop first", "free RAM"],
        "forbidden_outputs": ["live migration", "live-migrate"],
        "synonym_groups": [
            ["cold migration", "cold migrate", "offline migration", "non-live migration"],
            ["stop first", "power off first", "shutdown before migrating", "halt VM"],
            ["free RAM", "available memory", "free memory", "RAM available"],
        ],
        "raw_text": raw_text,
    }


def _build_delayed_dependency(
    idx: int, nova_lines: list[str], inst: str
) -> dict:
    req  = _reqid()
    port = _portid()
    neutron_warning = _fill(NEUTRON_TEMPLATES[3], inst=inst, req=req, port=port)
    neutron_sync    = _fill(NEUTRON_TEMPLATES[2], inst=inst, req=req)
    noise = _noise_lines(8, inst=inst, req=req)
    mid = max(2, len(nova_lines) // 2)
    all_lines = nova_lines[:mid] + [neutron_warning, neutron_sync] + noise[:4] + nova_lines[mid:] + noise[4:]
    all_lines = all_lines[:15]

    constraint = (
        f"CONSTRAINT: Neutron network [port: {port}] requires security group rule "
        f"sync before VM instance [{inst}] becomes reachable. "
        f"iptables rules must be flushed and re-applied before marking the VM ready."
    )
    raw_text = constraint + "\n" + "\n".join(all_lines)

    return {
        "family":          "delayed_dependency",
        "failure_mode":    "delayed_dependency_loss",
        "services":        ["nova", "neutron"],
        "question":        "What network prerequisite must be satisfied before this VM becomes reachable?",
        "required_outputs": ["security group", "sync", "iptables"],
        "forbidden_outputs": ["skip security", "bypass"],
        "synonym_groups": [
            ["security group", "secgroup", "firewall rule", "network policy"],
            ["sync", "synchronize", "flush", "re-apply"],
            ["iptables", "netfilter", "packet filter", "firewall"],
        ],
        "raw_text": raw_text,
    }


def _build_dependency_chain(
    idx: int, nova_lines: list[str], inst: str
) -> dict:
    req  = _reqid()
    vol  = _volid()
    port = _portid()
    ks_line  = _fill(KEYSTONE_TEMPLATES[0], req=req)
    nova_sched = _fill(NOVA_TEMPLATES[6], inst=inst)
    neu_line = _fill(NEUTRON_TEMPLATES[0], inst=inst, req=req, port=port)
    cin_line = _fill(CINDER_TEMPLATES[1], inst=inst, vol=vol)
    noise = _noise_lines(8, inst=inst, req=req)
    # Inject the 4 chain steps prominently, interleaved with noise
    mid = max(1, len(nova_lines) // 3)
    all_lines = (
        [ks_line] + nova_lines[:mid] + [nova_sched, neu_line] +
        noise[:3] + nova_lines[mid:] + [cin_line] + noise[3:7]
    )
    all_lines = all_lines[:15]

    constraint = (
        f"CONSTRAINT: OpenStack VM launch sequence for instance [{inst}] "
        f"must follow: Step 1 Keystone auth, Step 2 Nova schedule, "
        f"Step 3 Neutron port allocation, Step 4 Cinder volume attach."
    )
    raw_text = constraint + "\n" + "\n".join(all_lines)

    return {
        "family":          "dependency_chain",
        "failure_mode":    "chain_step_loss",
        "services":        ["nova", "neutron", "keystone", "cinder"],
        "question":        "List all OpenStack services that must complete successfully in order.",
        "required_outputs": ["Keystone", "Nova", "Neutron", "Cinder"],
        "forbidden_outputs": [],
        "synonym_groups": [
            ["Keystone", "auth service", "identity service", "token provider"],
            ["Neutron", "network service", "networking", "SDN controller"],
            ["Cinder", "block storage", "volume service", "storage backend"],
        ],
        "raw_text": raw_text,
    }


def _build_false_distractor(
    idx: int, nova_lines: list[str], inst: str
) -> dict:
    req = _reqid()
    img = _imgid()
    glance_error = _fill(GLANCE_TEMPLATES[2], img=img, req=req, inst=inst)
    glance_warn  = _fill(GLANCE_TEMPLATES[4], img=img, req=req)
    noise = _noise_lines(8, inst=inst, req=req)
    mid = max(2, len(nova_lines) // 2)
    all_lines = nova_lines[:mid] + [glance_error, glance_warn] + noise[:4] + nova_lines[mid:] + noise[4:]
    all_lines = all_lines[:15]

    constraint = (
        f"CONSTRAINT: Glance image [{img}] download errors must NOT be treated "
        f"as Nova scheduler failures. The root cause is in the Glance storage backend, "
        f"not the nova-scheduler or compute host selection."
    )
    raw_text = constraint + "\n" + "\n".join(all_lines)

    return {
        "family":          "false_distractor",
        "failure_mode":    "false_archive",
        "services":        ["nova", "glance"],
        "question":        "Which service is the true root cause, ignoring the nova-scheduler errors?",
        "required_outputs": ["image", "Glance", "storage backend"],
        "forbidden_outputs": ["scheduler", "nova-scheduler", "compute host"],
        "synonym_groups": [
            ["Glance", "image service", "image registry", "glance store"],
            ["storage backend", "image store", "glance store", "object store"],
            ["image", "VM image", "boot image", "glance image"],
        ],
        "raw_text": raw_text,
    }


def _build_negation_critical(
    idx: int, nova_lines: list[str], inst: str
) -> dict:
    req  = _reqid()
    user = _user()
    ks_fail  = _fill(KEYSTONE_TEMPLATES[3], user=user, req=req)
    ks_start = _fill(KEYSTONE_TEMPLATES[0], user=user, req=req)
    noise = _noise_lines(8, inst=inst, req=req)
    mid = max(2, len(nova_lines) // 2)
    all_lines = nova_lines[:mid] + [ks_fail, ks_start] + noise[:4] + nova_lines[mid:] + noise[4:]
    all_lines = all_lines[:15]

    constraint = (
        f"CONSTRAINT: Keystone token validation failures for user [{user}] "
        f"must NOT be auto-renewed without explicit user re-authentication. "
        f"Automatic token refresh is prohibited — the user must manually re-authenticate."
    )
    raw_text = constraint + "\n" + "\n".join(all_lines)

    return {
        "family":          "negation_critical",
        "failure_mode":    "negation_loss",
        "services":        ["keystone", "nova"],
        "question":        "What action is explicitly prohibited for this token/auth failure?",
        "required_outputs": ["re-authenticate", "user action", "manual"],
        "forbidden_outputs": ["auto-renew", "automatic token", "silent refresh"],
        "synonym_groups": [
            ["re-authenticate", "re-auth", "login again", "re-login", "new authentication"],
            ["manual", "explicit", "user-initiated", "operator action"],
            ["user action", "human intervention", "user must", "operator must"],
        ],
        "raw_text": raw_text,
    }


def _build_synonym_match(
    idx: int, nova_lines: list[str], inst: str
) -> dict:
    req  = _reqid()
    proj = _proj()
    nova_rate = (
        f"[nova.api.openstack.wsgi] [WARNING] Rate limit exceeded for project "
        f"{proj} req_id={req} — HTTP 429 Too Many Requests"
    )
    noise = _noise_lines(8, inst=inst, req=req)
    mid = max(2, len(nova_lines) // 2)
    all_lines = nova_lines[:mid] + [nova_rate] + noise[:5] + nova_lines[mid:] + noise[5:]
    all_lines = all_lines[:15]

    constraint = (
        f"CONSTRAINT: Nova API rate limit violations for project [{proj}] "
        f"must be reported to the billing system for usage accounting and charge-back."
    )
    raw_text = constraint + "\n" + "\n".join(all_lines)

    return {
        "family":          "synonym_match",
        "failure_mode":    "synonym_loss",
        "services":        ["nova", "neutron"],
        "question":        "What external system must be notified of this rate limit violation?",
        "required_outputs": ["billing"],
        "forbidden_outputs": [],
        "synonym_groups": [
            ["billing", "charge", "accounting", "metering", "usage report",
             "charge-back", "chargeback", "cost tracking"],
        ],
        "raw_text": raw_text,
    }


FAMILY_BUILDERS = [
    ("hard_constraint",     _build_hard_constraint),
    ("delayed_dependency",  _build_delayed_dependency),
    ("dependency_chain",    _build_dependency_chain),
    ("false_distractor",    _build_false_distractor),
    ("negation_critical",   _build_negation_critical),
    ("synonym_match",       _build_synonym_match),
]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_all(rows: list[dict]) -> list[dict]:
    """Generate N_PER_FAMILY × 6 incidents, interleaved by family."""
    # Pick centers for all families
    total_needed = N_PER_FAMILY * len(FAMILY_BUILDERS)
    centers = _pick_centers(rows, total_needed, seed_offset=0)
    # Assign centers round-robin: family 0 gets centers 0,6,12,...
    n_fam = len(FAMILY_BUILDERS)
    family_centers: dict[str, list[int]] = {name: [] for name, _ in FAMILY_BUILDERS}
    for i, center in enumerate(centers):
        fname = FAMILY_BUILDERS[i % n_fam][0]
        family_centers[fname].append(center)

    # Ensure each family has exactly N_PER_FAMILY centers
    for fname, clist in family_centers.items():
        while len(clist) < N_PER_FAMILY:
            clist.append(random.randint(0, len(rows) - 1))
        family_centers[fname] = clist[:N_PER_FAMILY]

    incidents: list[dict] = []
    rng = random.Random(42)
    # Interleave: for each i in 0..N_PER_FAMILY-1, emit one incident per family
    for inc_idx in range(N_PER_FAMILY):
        for family_name, builder in FAMILY_BUILDERS:
            center     = family_centers[family_name][inc_idx]
            nova_lines = _nova_window(rows, center, before=3, after=6)
            inst       = _extract_instance_uuid(nova_lines)
            if not inst or inst == "":
                inst = _instid()

            inc_data = builder(inc_idx, nova_lines, inst)

            incident_num = inc_idx + 1
            inc = {
                "incident_id":      f"openstack_{family_name}_{incident_num:03d}",
                "family":           family_name,
                "failure_mode":     inc_data["failure_mode"],
                "source":           "openstack_loghub",
                "services":         inc_data["services"],
                "raw_text":         inc_data["raw_text"],
                "question":         inc_data["question"],
                "required_outputs": inc_data["required_outputs"],
                "forbidden_outputs": inc_data["forbidden_outputs"],
                "synonym_groups":   inc_data["synonym_groups"],
            }
            incidents.append(inc)

    total = N_PER_FAMILY * len(FAMILY_BUILDERS)
    assert len(incidents) == total, f"Expected {total}, got {len(incidents)}"
    return incidents


def _write_jsonl(path: str, incidents: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for inc in incidents:
            f.write(json.dumps(inc, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(incidents):>4} incidents -> {os.path.basename(path)}")


def main() -> None:
    rows = _load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} rows from {CSV_PATH}")

    levels = Counter(r["Level"] for r in rows)
    comps  = Counter(r["Component"] for r in rows)
    print(f"Level distribution: {dict(levels)}")
    print(f"Top 5 components: {dict(comps.most_common(5))}")

    incidents = generate_all(rows)
    total = len(incidents)
    print(f"\nGenerated {total} incidents ({N_PER_FAMILY} per family × {len(FAMILY_BUILDERS)} families)")

    # Write master list
    _write_jsonl(OUT_PATH, incidents)

    # Write train / val / test splits
    train_inc = incidents[:TRAIN_N]
    val_inc   = incidents[TRAIN_N:TRAIN_N + VAL_N]
    test_inc  = incidents[TRAIN_N + VAL_N:]
    assert len(test_inc) == TEST_N, f"Expected {TEST_N} test, got {len(test_inc)}"

    _write_jsonl(TRAIN_PATH, train_inc)
    _write_jsonl(VAL_PATH,   val_inc)
    _write_jsonl(TEST_PATH,  test_inc)

    # Validation
    fam_count = Counter(inc["family"] for inc in incidents)
    print(f"\nFamily breakdown (master): {dict(sorted(fam_count.items()))}")

    svc_count: Counter = Counter()
    for inc in incidents:
        for s in inc["services"]:
            svc_count[s] += 1
    print(f"Services represented: {dict(svc_count.most_common())}")

    # Sample first incident
    first = incidents[0]
    print(f"\nSample incident (first):")
    print(f"  id={first['incident_id']}  family={first['family']}")
    print(f"  required={first['required_outputs']}")
    print(f"  forbidden={first['forbidden_outputs']}")
    obs_count = first["raw_text"].count("\n")
    print(f"  raw_text lines: {obs_count + 1}")
    print(f"  raw_text preview: {first['raw_text'][:200]!r}")

    # Train/val/test family balance
    for split_name, split_inc in [("train", train_inc), ("val", val_inc), ("test", test_inc)]:
        fc = Counter(inc["family"] for inc in split_inc)
        print(f"  {split_name:5s} ({len(split_inc):3d}): {dict(sorted(fc.items()))}")


if __name__ == "__main__":
    main()
