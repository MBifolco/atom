
# 🗂️ Atom Combat — Registry (Component)

## Purpose
The **Registry** is the official catalog of everything allowed to exist in Atom Combat: fighters, arenas, world specs, and protocol versions.

> The Registry is the sport’s memory and gatekeeper.

---

## Responsibilities
- Store all certified fighters, arenas, and league data.  
- Manage versioning, eligibility, and deprecations.  
- Track lineage: creator, checksum, validation status.  
- Support league organization and tournament enrollment.  

---

## Example Registry Record
```json
{
  "id": "fighter_aeon_v3",
  "creator": "Team Nova",
  "division": "pro",
  "protocol_version": "proto_v2",
  "world_spec": "world_v3",
  "certified": true,
  "checksum": "1f2b9a",
  "last_updated": "2025-11-03"
}
```

---

## Structure
| Category | Contents |
|-----------|-----------|
| **Fighters** | Certified fighter artifacts |
| **Arenas** | Approved world environments |
| **World Specs** | Active division rulesets |
| **Protocols** | Contract definitions |
| **Seasons** | Historical results and patches |

---

**One-liner summary:**  
*The Registry is the trusted archive of Atom Combat — the ledger that keeps every fighter and rule version accountable.*
