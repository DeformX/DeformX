# GPU candidate validation · 2026-11-08

The candidate pins CUDA backend `96b8514c7c55c724f1abba1fbf6ed6642b7db72b`. Validation ran on an RTX 5090 with CUDA 12.8.93, Python 3.11.15, PyTorch 2.7.0+cu128, NumPy 1.26.0, GCC 11.4, CMake 3.31.8 and GLM 0.9.9.8.

| Check | Observed result |
|---|---|
| CUDA backend tests | 134 passed, 1 strict expected failure |
| Parent example tests | 73 passed, no skips |
| Five basic CLI scenes, default settings | all complete; all case criteria passed |
| Tip-target CLI, default feedback policy | complete; target criterion passed |
| Braider, default short preset | complete; all ten gates passed |
| Traction reels, default one-cycle run | complete; transport criteria passed |
| Arena, 64 worlds, 2 s measured after 2 s settling | complete; contact criteria passed |

[Machine-readable settings, commands and source hashes](../demo/validation/gpu-release-checks.json) · [Parent test log](../demo/validation/parent-tests.log) · [Backend test log](../demo/validation/backend-tests.log)

The expected failure documents a stiff-tube regime (`EI = 10 N·m²`) that becomes unstable during root reorientation at its selected resolution. It is not hidden or counted as a pass.

The loaded extension's SHA-256 was `652156b9af5a4b249dc6a1ed007894e415b438322ad771e5a14ad18efbbc4f85`. Its path and linked CUDA library were checked before accepting the suite result. The final source was rebuilt after documentation edits; the extension remained byte-for-byte identical to the tested binary.

Backend changes also passed the relative-output-path regression and incomplete-recording checks for its new showcase scripts. Short three-frame twist and net runs verified startup and export only; they do not establish full-duration winding, bunny catch or settling behavior. The complete asset-driven Shadow Hand example was not run in this validation.

The bundled videos replay the original simulation trajectories and retain their timings. After the numerical validation, the presentation was refreshed without text overlays: existing RGB captures were re-encoded for the basics, and the braider, reels, guitar and arena were re-rendered from saved states. This did not repeat the physics runs or the full 1,024-world performance recording. The braider's reference refinement limitation remains documented in [its README](../demo/braiding_machine/README.md).

The presentation-only `widen_frame.py` was separately executed in Blender 4.5.9 on the server. Its [asset manifest](../demo/braiding_machine/media/wide-frame-assets.json) records the approximately 2.10 m × 3.16 m frame, unchanged grip and carrier-path geometry, and byte-identical moving carrier assets. This geometry revision changes the display structure, not the rope model or recorded trajectories.

The added rack-insertion and cascade videos are archived physical trials replayed with clean RGB captures. They have separate reports and trajectory hashes in their example folders; they are not additional runs of the nine-case release check above. The rack clip skips acquisition and retains the original contact-held insertion and release. Its task outcome does not establish converged contact-force prediction.

The new [Panda cable extraction](../demo/cable_extraction/README.md) is a separate physical run on the pinned candidate, not an archived replay-only addition. The longer version uses 193 physical vertices per strand, a roughly 1.25 m target and four 1.68–1.77 m neighboring cables. Its 37.4 s recording passed the independent separation/contact audit. It uses prescribed arm goals and known placement, with a free rigid pull tab held by actual finger contact. The example records evidence for the exact run, matched open-gripper control and failed route trials; raw trajectories remain on the validation host. Its changed layout, material/contact assumptions and solver settings are documented; this is not a resolution-convergence comparison against the shorter version.

The combined preview is one 21-second, two-column, three-row GIF/MP4: humanoid guitar and braiding on the first row, cable extraction and rack insertion on the second, traction reels and the contact arena on the third. Braiding plays its complete recording at 1×, with all 630 source frames checked in order against the uncompressed composite panel after scaling. Extraction now includes its full 18.7-second accelerated recording from the beginning and holds the last frame for 2.3 seconds. The other four panels repeat their eight-second excerpts; the standalone braider GIF also covers the full 21 seconds. Guitar opens with 2 s of pick close-up and about 1 s of pullback at 0.5× simulation speed, then about 5 s of full-body performance at 1×. Extraction plays at 2×; braiding, rack insertion, traction reels and the arena play at 1×. The cascade remains linked as a full video. Extraction's original 37.4 s capture is also retained. Its 18.7 s presentation preserves all 1,122 encoded frames at 60 fps with packet timestamps and durations halved. The composition and individual animated previews were encoded and fully decoded on the server. Their source trims, playback rates, per-scene evidence and output hashes are recorded in `demo/media/`. The [guitar preview frame map](../demo/humanoid_guitar/presentation.json) also verifies all 240 selected source frames against the lossless edit before encoding, without motion interpolation or new physics.

The braider's [camera audit](../demo/braiding_machine/render-stability-hold.json) verifies a five-second close-up at seconds 4–9 where the strands converge and form the braid. The camera eye and target are 0.70 m higher than in the previous hold. The timing, opening and whole-machine views and studio lighting match the previous replay. The audit verifies all 630 original states and timestamps at 1× and checks fixed-backdrop brightness in the native frames and encoded video.

The arena's [near-to-wide replay audit](../demo/contact_arena/render-stability-pullback.json) independently verifies all 720 archived states and 30 fps timestamps, zero physics-clock advance, and full-array camera bounds for the 555-frame wide hold. The camera holds near for 1.5 s, pulls back until 5.5 s and holds wide through 24 s. It explicitly selects the original Blender HDR environment and matches the archived camera's appearance within 1.04 RGB code values mean absolute error. Camera-tracked plate patches show coherent adjacent brightness changes below 0.35 luma codes; [local excursions](../demo/contact_arena/render-stability-pullback-local.json) remain, so this is not a whole-image flicker guarantee. The standalone arena GIF includes all 24 seconds; its hero panel repeats seconds 0–8, including the entire pullback.

The subsequent [rendering stability review](../demo/media/render-stability-review.json) audits all six scenes and supplies validated native DLAA recordings for rack insertion and traction reels. All 42 files in the earlier release-check source manifest remain identical. The traction-reel experimental capture passed full pose, appearance and timing checks, but its initialization sensitivity remains unresolved; its portable renderer was therefore retained. The numerical models, archived trajectories and backend pin are unchanged; the earlier numerical suite was not rerun for these presentation changes.

All CUDA builds, numerical execution and media composition for this candidate ran on the designated server. Original experiment checkouts and installed runtimes were left in place.

## Portable showcase validation · 2026-11-09

Guitar, rack insertion and cable cascade now include asset preparation where needed, physical runners, audits and native renderers. Each was run from newly generated state on `ps` with the same pinned release extension, followed by native rendering and full video decode. [Commands, source hashes and suite evidence](../demo/validation/portable-examples.json) · [108 passing parent tests, no skips](../demo/validation/portable-parent-tests.log).

| Example | Fresh physical result | Fresh native video |
|---|---|---|
| [Guitar](../demo/humanoid_guitar/evidence/portable-validation.json) | 12.45 s; all 120 ordered pick contacts across 24 strokes; continuous visible fingertip/fret support | 478 frames, 15.93 s edited playback |
| [Rack insertion](../demo/rack_insertion/evidence/portable-validation.json) | 20.07 s; acquisition and insertion passed; retained 1.30 s after 119.94 mm withdrawal | 602 frames, 20.07 s at 1× |
| [Cable cascade](../demo/cable_cascade/evidence/portable-validation.json) | 48 rods, 6 s, 720 steps; finite states and no flags; 2.238% peak strain | 181 frames, 6.03 s including final state |

Guitar's 63 upstream inputs and rack's 42 pinned files were downloaded and checked; preparation did not depend on the old experiment assets. Rack also uses the documented NVIDIA grid environment. Its first launch failed from importing standalone Warp before Kit; the corrected worker loads bundled Warp 1.8.2, and its supervisor rejects incomplete runs even if Kit returns exit 0. The initial failure remains in the evidence.

These are configured demonstration checks, not general manipulation policies or force-convergence studies. Cascade's smallest sampled rod/rod and sphere surface gaps were −0.787 mm and −0.532 mm. The original nine examples' physics source and backend pin are unchanged; the earlier catalog test was extended for the three new entries.

The updated hero starts extraction at source time zero, plays its full 18.7 s at 2× and holds the true final source frame for 2.3 s. Its [frame audit](../demo/media/gpu-showcase-provenance.json) independently checks all 561 sampled extraction frames, both original endpoints, the final hold and all 630 braider frames. Obsolete hero exports were removed; the README links the latest content-addressed GIF and MP4.
