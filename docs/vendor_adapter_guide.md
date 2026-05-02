# Vendor Adapter Guide

> Adding support for a new AOI maker = ~300 LOC + one test fixture.

## Contract

Every adapter implements `VendorAdapter` (`aoi_sentinel/adapters/base.py`):

```python
class VendorAdapter(Protocol):
    name: str
    def watch(self, source) -> Iterator[CommonInspection]: ...
    def push_verdict(self, board_id, verdicts) -> None: ...
```

`CommonInspection` is the only schema the core engine ever sees. Whatever the
vendor's native format looks like — XML, CSV, OPC-UA, REST, vendor SDK — the
adapter normalises it.

## Schema

```python
@dataclass
class CommonInspection:
    board_id: str
    timestamp: datetime
    vendor: str            # adapter.name
    line_id: str | None
    lot: str | None
    components: list[ComponentInspection]
    raw_payload_path: str | None

@dataclass
class ComponentInspection:
    ref_des: str
    bbox_xyxy: tuple[int, int, int, int]
    image_2d: np.ndarray            # HWC uint8
    height_map: np.ndarray | None   # 2D float32 (mm or microns), if 3D AOI
    vendor_call: "PASS" | "DEFECT" | "UNKNOWN"
    vendor_defect_type: str | None  # vendor's own defect taxonomy
    extra: dict
```

## Implementing a new adapter

1. Create `aoi_sentinel/adapters/<vendor>/__init__.py`.
2. Class with `@register("<vendor>")` decorator.
3. `watch(source)` is a generator — yield one `CommonInspection` per board.
4. Be **resilient**: malformed files must be logged and skipped, never crash.
5. `push_verdict` may be a no-op + log if there's no return channel.
6. Add a fixture (`tests/fixtures/<vendor>_sample.zip`) and tests.

## Source kinds we already handle

| Kind | Watch strategy | Example adapters |
|------|---------------|-------------------|
| File share (SMB/NFS) | `Path.rglob` poll | `saki`, `koh_young` (KSDB export), `generic_csv` |
| REST/long-poll | `httpx` async loop | `koh_young` (K-API, planned) |
| OPC-UA | `asyncua` subscription | (planned for Mycronic / TRI integrations) |
| Vendor SDK callback | adapter-side callback registration | (per vendor) |

## Per-vendor notes

### Saki (BF-3Di / 3Di-LU / SD-Series)

- Output: `*.xml` per board + image folder (2D top, 2D side, 3D height map TIFF).
- XML schema differs across **PowerScout** and **PowerView** versions — confirm with site IT before deploying.
- Image paths inside the XML are relative to the XML directory.
- **Status**: skeleton in `adapters/saki/`; complete implementation pending a real sample.

### Koh Young (Zenith / Aspire / Neptune)

- Primary path: KSDB-Studio CSV/XML export to a watched folder.
- Expected columns (from public KSMART screenshots — verify on first deploy):

  ```
  Barcode, InspectionTime, LineName, LotNo,
  RefDesignator, X, Y, Width, Height,
  Image2D, Image3D, Judgement, DefectCode
  ```

- Secondary path (`backend="k_api"`): K-API REST, per-customer SDK contract.
- **Avoid**: direct DB read from KSDB Postgres/MSSQL — vendor doesn't bless it.
- **Status**: skeleton in `adapters/koh_young/`; CSV parser pending real sample.

### generic_csv (fallback / smoke test)

- For any AOI that exports a result folder + image folder + a CSV index.
- Schema is documented in the module docstring.
- Useful for benchmarking on public datasets and for vendors we haven't written a dedicated adapter for yet.

## Verdict back-channel — phased

| Phase | Behaviour |
|-------|-----------|
| 1 | adapter logs verdicts only; nothing pushed to vendor system |
| 2 | adapter writes a sidecar JSON next to the original payload, MES picks up via existing share watch |
| 3 | adapter pushes via vendor SDK / API directly into MES (per-vendor) |

Always start at Phase 1 in SHADOW mode; promote per the safety gate.
