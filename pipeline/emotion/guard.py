"""Deterministic delivery guard.

A 0.8B model will occasionally tag a data-loss turn as `[emotion:warm]`, or
slip a `[chuckle]` into a security warning. That is not a quality nit: the
product rule is that data loss, security and irreversible actions are never
joked about. Prompting alone cannot guarantee it, so this module enforces it
outside the model.

Two jobs:

1. Recognize a serious user turn from the words themselves (destructive
   commands, data loss, credentials, security, power state).
2. Rewrite the delivery so a serious turn is `[emotion:serious]` with every
   non-verbal removed, and a normal turn still has a valid delivery tag.

The guard is deliberately conservative. A false positive costs a warmer reply
in a serious tone; a false negative costs the product's credibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DELIVERY_RE = re.compile(r"\[emotion:(neutral|amused|warm|focused|serious)\]")
NONVERBAL_RE = re.compile(r"\[(laugh|chuckle|sigh|gasp|hm|erm)\]")
BARE_TAG_RE = re.compile(
    r"\[(neutral|amused|warm|focused|serious|laugh|chuckle|sigh|gasp|hm|erm)\]"
)
CONFIRM_HINT_RE = re.compile(r"\b(confirm|sure|proceed|go ahead|shall i|should i|want me to)\b", re.I)

# Hard evidence that the turn is about damage, secrets or an irreversible
# action. These do not depend on the model's judgement.
SERIOUS_PATTERNS: tuple[tuple[str, str], ...] = (
    # destructive commands
    (r"\brm\s+(-[a-z]*\s+)*-[a-z]*[rf]", "destructive:rm"),
    (r"\brm\s+(-[a-z]*\s+)*-(?:r|f)", "destructive:rm"),
    (r"\bdd\s+(?:if|of)=", "destructive:dd"),
    (r"\bmkfs(?:\.\w+)?\b", "destructive:mkfs"),
    (r"\bshred\b", "destructive:shred"),
    (r"\bwipefs\b", "destructive:wipefs"),
    (r"\btruncate\b", "destructive:truncate"),
    (r"\bchmod\s+(?:-R\s+)?777\b", "security:chmod-777"),
    (r"\bchown\s+-R\b", "security:chown"),
    (r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|fi)?sh\b", "security:pipe-to-shell"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "destructive:fork-bomb"),
    # data loss
    (r"\b(?:lost|losing|deleted|gone|wiped|erased|corrupted|formatted|overwrote|destroyed)\b"
     r"[^.\n]{0,40}\b(?:files?|data|project|drive|disk|photos?|documents?|backup|partition|folder)\b",
     "data-loss"),
    (r"\b(?:files?|data|project|drive|disk|photos?|documents?|backup|partition|folder)\b"
     r"[^.\n]{0,40}\b(?:lost|deleted|gone|wiped|erased|corrupted|formatted|overwrote|destroyed)\b",
     "data-loss"),
    (r"\bno backup\b|\bnot backed up\b|\bwithout a backup\b", "data-loss:no-backup"),
    # credentials and security
    # A private key is only a serious turn when it is exposed or lost. "Save
    # my SSH key" is a request, not an incident.
    (r"(?:(?:ssh|gpg)\s+(?:private\s+)?key|private\s+(?:ssh\s+)?key|(?:secret|signing)\s+key)"
     r"[^.\n]{0,40}\b(?:public|committed|leak(?:ed)?|exposed|lost|stolen|copied|shared|posted|pasted|printed|repo|gist|chat|log|compromised|same disk)\b",
     "security:private-key"),
    (r"\b(?:public|committed|leak(?:ed)?|exposed|lost|stolen|copied|shared|posted|pasted|printed|repo|gist|chat|log|compromised|same disk)\b"
     r"[^.\n]{0,40}(?:(?:ssh|gpg)\s+(?:private\s+)?key|private\s+(?:ssh\s+)?key|(?:secret|signing)\s+key)",
     "security:private-key"),
    (r"\b(?:password|passphrase|credential|token)s?\b[^.\n]{0,40}"
     r"\b(?:leaked|stolen|exposed|compromised|shared|plaintext|in the clear|written down)\b",
     "security:credential-leak"),
    (r"\b(?:phishing|malware|ransomware|virus|breach(?:ed|es|ing)?|hacked|compromised|keylogger)\b",
     "security:compromise"),
    (r"\b(?:sudo|root)\b[^.\n]{0,40}\b(?:password|access|permission|privilege)\b",
     "security:privilege"),
    # irreversible system actions
    (r"\b(?:factory reset|reinstall the (?:os|system)|format the (?:disk|drive|partition))\b",
     "irreversible:reset"),
    (r"\b(?:shutdown|reboot|poweroff|halt|suspend(?:ing|ed)?)\b", "irreversible:power"),
    (r"\bkillall\b|\bkill\s+-9\b|\bpkill\b|\bsystemctl\s+(?:stop|disable|mask)\b",
     "irreversible:service"),
    (r"\b(?:rm|delete|remove|wipe|erase)\b[^.\n]{0,30}"
     r"\b(?:everything|all files|the whole|entire|system|root|home directory)\b",
     "destructive:bulk"),
    # storage failure
    (r"\bdisk (?:full|failure|error)\b|\bno space left\b|\bbad sectors?\b|"
     r"\bsmart (?:error|failure)\b|\bdrive (?:is )?(?:failing|dying)\b",
     "storage:failure"),
)

# Patterns added after the first probe round. Each one exists because a real
# prompt slipped through: boot failure, an unfamiliar login, a lost key, a
# firewall being disabled. Keep this list boring and specific.
SERIOUS_PATTERNS += (
    (r"\bdisk\s+is\s+full\b", "storage:failure"),
    (r"\bdelet(?:e|ed|ing)\b[^.\n]{0,30}\b(?:database|repo|repository|branch|commits?|history|home directory|home folder)\b",
     "data-loss"),
    (r"\bkill\s+all\b", "irreversible:service"),
    (r"\bwon'?t boot\b|\bwill not boot\b|\bunbootable\b", "storage:boot"),
    (r"(?:not|never|isn'?t|aren'?t)\s+backed up\b|\bwithout\s+(?:a\s+)?backup\b", "data-loss:no-backup"),
    (r"\b(?:grinding|clicking)\b[^.\n]{0,20}\b(?:drive|disk)\b", "storage:failure"),
    (r"\b(?:drive|disk)\b[^.\n]{0,20}\b(?:grinding|clicking)\b", "storage:failure"),
    (r"\bwipe\b[^.\n]{0,20}\b(?:laptop|machine|computer|phone|drive|disk)\b", "destructive:device"),
    # Disabling a firewall is serious when the user says why it is risky
    # (public network, demo, forgot to re-enable) or when it is blocking
    # everything and they are about to turn it off. A bare "it's annoying"
    # is a policy/confirmation turn, not a serious-affect turn.
    (r"\b(?:disabl(?:e|ed|ing)|turn(?:ed|ing)?\s+off)\b[^.\n]{0,25}\bfirewall\b"
     r"[^.\n]{0,60}\b(?:public|demo|forgot|forget|forgotten|still off|exposed|while it was off|blocking everything|blocking my)\b",
     "security:firewall"),
    (r"\bfirewall\b[^.\n]{0,45}\bblocking\s+(?:everything|my)\b[^.\n]{0,70}\b(?:turn(?:ed|ing)?\s+(?:it\s+)?off|disabl(?:e|ing))\b",
     "security:firewall"),
    (r"\b(?:someone|anyone|unknown|unauthorized)\b[^.\n]{0,30}\b(?:logged in|logged into|session|connected)\b",
     "security:unauthorized-access"),
    (r"\b(?:aws|api|access)\s+(?:key|token|secret)\b", "security:secret"),
    (r"\b(?:lost|misplaced)\b[^.\n]{0,25}\b(?:encryption|private|ssh|gpg)\s+key\b", "security:private-key"),
    (r"\buninstall\b[^.\n]{0,25}\b(?:kernel|bootloader|display manager)\b", "irreversible:uninstall"),
    (r"\bshut(?:ting)?\s+down\b", "irreversible:power"),
    (r"\braid\b[^.\n]{0,30}\bdegraded\b|\bdegraded\b[^.\n]{0,20}\braid\b", "storage:raid"),
    (r"\b(?:drive|disk)\b[^.\n]{0,20}\bunallocated\b", "storage:partition"),
    (r"\bdelet(?:e|ed|ing)\b[^.\n]{0,30}\b(?:backups?|drive|disk|partition|files?|folder|data)\b", "data-loss"),
    (r"\b(?:curl|wget)\b[^.\n]{0,40}\b(?:ba|z)?sh\b", "security:pipe-to-shell"),
    (r"\b(?:database|repo|repository|filesystem)\b[^.\n]{0,25}\b(?:corrupted|damaged|inconsistent)\b", "data-loss:corruption"),
    (r"\b(?:corrupted|damaged)\b[^.\n]{0,25}\b(?:database|repo|repository|filesystem)\b", "data-loss:corruption"),
    (r"\bformat\s+/dev/", "destructive:mkfs"),
    (r"\b(?:delet(?:e|ed|ing)|remov(?:e|ed|ing)|rm)\b\s+/(?:etc|usr|bin|boot|dev|var|home|root)\b", "destructive:system-path"),
    (r"/etc/(?:shadow|gshadow)\b|\.ssh/id_\w+|\.aws/credentials\b", "security:credential-file"),
    (r"\b(?:delet(?:e|ed|ing)|remov(?:e|ed|ing)|rm|wipe)\b[^.\n]{0,30}\bhome\s+(?:folder|directory)\b",
     "destructive:home"),
    (r"\bdd\b[^.\n]{0,25}\b(?:over|onto|to)\b[^.\n]{0,25}\b(?:drive|disk|partition)\b", "destructive:dd"),
    (r"\b(?:overwrote|overwritten|overwrite)\b", "data-loss:overwrite"),
    (r"\braid\b[^.\n]{0,30}\b(?:failed|failure|degraded)\b|\b(?:failed|degraded)\b[^.\n]{0,20}\braid\b", "storage:raid"),
    (r"\bchown'?d?\b[^.\n]{0,30}\b(?:system|whole|root|/)\b", "security:chown"),
    (r"\b(?:lost|stolen|missing)\b[^.\n]{0,20}\b(?:phone|laptop|device)\b", "security:device"),
    (r"(?<!\w)\.env\b[^.\n]{0,40}\b(?:repo|git|public|committed|password|secret)\b", "security:env"),
    (r"\b(?:sudo|root)\b[^.\n]{0,30}\b(?:random|unknown|untrusted|website|site|script)\b", "security:privilege"),
    (r"\bgit\s+reset\s+--hard\b|\blost\b[^.\n]{0,20}\buncommitted\b", "data-loss:git"),
    (r"\bdrop(?:ped|ping)?\b[^.\n]{0,20}\b(?:production|prod|table|database)\b", "data-loss:database"),
    (r"\bencryption\s+key\b[^.\n]{0,35}\b(?:backup|same disk|public|lost|exposed)\b", "security:private-key"),
    (r"\bpower\s+(?:went out|cut|failure|loss|outage)\b", "storage:power"),
    (r"\b(?:give|share|send|tell|hand)\b[^.\n]{0,25}\b(?:password|passphrase|secret)\b", "security:credential-leak"),
    (r"\bsecret(?:s)?\b[^.\n]{0,35}\b(?:plaintext|committed|public|leaked|exposed|in the clear)\b", "security:secret"),
    (r"\b(?:random|unknown|untrusted)\b[^.\n]{0,35}\b(?:command|script|website|site|download)\b[^.\n]{0,25}\bsudo\b", "security:privilege"),
    # v3 clean-probe round. These are ordinary phrasings that slipped through
    # the first list; each one is a real incident, not a style preference.
    (r"\b(?:phone|laptop|device)\b[^.\n]{0,25}\b(?:lost|stolen|missing)\b", "security:device"),
    (r"\braid\b[^.\n]{0,80}\b(?:failed|failure|degraded|rebuilding)\b", "storage:raid"),
    (r"\b(?:only|last|sole)\s+backup\b", "data-loss:no-backup"),
    (r"\bbackup\s+(?:drive|disk)\b[^.\n]{0,40}\b(?:dropped|failed|failing|no longer|not mount|unmount|missing)\b",
     "storage:backup"),
    (r"\bauthorized_keys\b", "security:authorized-keys"),
    (r"\b(?:database|replication|replica)\b[^.\n]{0,40}\bfail\s*over\b|\bfail\s*over\b[^.\n]{0,40}\b(?:database|replication|replica)\b",
     "data-loss:database"),
    (r"\bencryption\s+key\b[^.\n]{0,60}\b(?:backup|same (?:folder|disk|drive)|public|lost|exposed)\b",
     "security:private-key"),
    (r"\b(?:disabl(?:e|ed|ing)|turn(?:ed|ing)?\s+off)\b[^.\n]{0,25}\bfirewall\b[^.\n]{0,60}\b(?:test|testing|temporarily|for a test)\b",
     "security:firewall"),
)

SECRET_READ_PATTERNS = (
    r"~/\.ssh/id_",
    r"/etc/shadow",
    r"~/.aws/credentials",
    r"~/.kube/config",
)


@dataclass(frozen=True)
class GuardResult:
    serious: bool
    reasons: tuple[str, ...]
    text: str


def serious_reasons(user_text: str) -> list[str]:
    """Every reason the turn must be handled seriously, in match order."""
    reasons: list[str] = []
    text = user_text or ""
    # Quoted page/email content is data, not the user's intent. A summarise
    # request should not become a data-loss turn because the page says
    # "delete all user files".
    text = re.sub(r"\[[^\]\n]{0,1000}\]", " ", text)
    # A page that merely *talks about* a destructive instruction is not a
    # request to perform it. This is deliberately narrow: jokes about data
    # loss are still caught by the patterns above.
    if re.search(r"\b(?:funny|silly|joke)\s+instruction\b", text, re.I):
        return []
    for pattern, reason in SERIOUS_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and reason not in reasons:
            reasons.append(reason)
    return reasons


def _strip_nonverbals(text: str) -> str:
    return NONVERBAL_RE.sub("", text)


def enforce(text: str, serious: bool) -> str:
    """Rewrite the delivery tag and remove non-verbals when serious."""
    out = _strip_nonverbals(text or "")
    if serious:
        out = DELIVERY_RE.sub("", out)
        return "[emotion:serious]" + out.strip()
    out = BARE_TAG_RE.sub("", out).strip()
    if not DELIVERY_RE.search(out):
        out = "[emotion:neutral]" + out
    return out


def apply(
    user_text: str,
    model_text: str,
    *,
    requires_confirmation: bool = False,
) -> GuardResult:
    """Classify the turn and return the corrected reply.

    `requires_confirmation` comes from the policy layer. When a destructive
    call is pending and the model forgot to ask, the guard appends the
    question rather than letting the runtime speak a reply that sounds like
    the action already happened.
    """
    reasons = serious_reasons(user_text)
    serious = bool(reasons)
    text = enforce(model_text, serious)
    if serious and requires_confirmation and not CONFIRM_HINT_RE.search(text):
        text = text.rstrip() + " Confirm?"
    return GuardResult(serious=serious, reasons=tuple(reasons), text=text)


def looks_serious(text: str) -> bool:
    return bool(serious_reasons(text))
