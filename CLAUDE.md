# mlx-swift-tipsv2 — agent notes

## Architecture

This package ships one library and two frontends that **share a single driver**:

- `Sources/TIPS` (`MLXTIPS`) — the model + the shared `TIPSSession` driver.
  All non-presentation work (model loading, inference, `CGImage` rendering via
  `TIPSRender`, HF download via `TIPSHub`) lives here.
- `Examples/tips-cli` — argument-parser CLI driving `TIPSSession`.
- `Examples/TIPSv2Demo` — SwiftUI app (standalone `.xcodeproj`) driving the
  same `TIPSSession`.

**Rule:** if you're deciding where code goes between the CLI and the GUI, the
answer is almost always "in `TIPSSession`/`TIPSRender`, not either frontend."
The frontends own only argument parsing / file pickers, cadence, and
`CGImage` → `NSImage`/PNG conversion. Never leak `NSImage`/SwiftUI types into
the library (`CGImage` is the presentation-neutral currency).

## Documentation

`MLXTIPS` ships DocC-generated reference docs (see
`Sources/TIPS/Documentation.docc/` and `scripts/build_docs.sh`).
**`///` doc comments on public/`open` symbols are published** to the static
site at https://mnmly.github.io/mlx-swift-tips/ and (if `EMIT_LLMS_TXT=1` is
used) into `docs/llms.txt`.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if the
  *why* is non-obvious. Skip restating what the signature already says.
- Document each parameter with `- Parameter name:` (use the **internal** name
  when there's an external label — DocC warns otherwise).
- Cross-reference related symbols with double-backtick links, e.g.
  `` ``TIPSSession/depthAndNormals(_:size:)`` ``. DocC link syntax is
  signature-sensitive: `foo(_:)` and `foo(_:_:)` are different.
- When you add a new top-level symbol that belongs in the curated sidebar, add
  it under the appropriate `## Topics` group in
  `Sources/TIPS/Documentation.docc/MLXTIPS.md`. Topics are organized by *user
  task*, not alphabetic order.

Verify before declaring documentation work done:

```bash
scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" or "external name used to document
parameter" warnings attributable to your changes.
