# Changelog

All notable changes to this project are documented in this file.

## [1.0.5] - 2026-04-03

### Added
- Added release hardening notes and versioned release entry for the completed sprint.
- Added camera readiness guard documentation in ActiveCamera lifecycle handling.

### Changed
- Unified camera runtime remains route-driven through ActiveCamera with mode values:
  - wandering
  - destination
- Reduced runtime logging noise in camera and YOLO inference paths to improve production performance.
- Updated voice-listening cleanup in Wayfinding to clear auto-restart timers on cleanup.
- Improved Choice voice listener setup to remove any previous listener before registering a new callback.
- Updated technical reference documentation to reflect current source code truth.

### Fixed
- Fixed repeated camera-ready setup behavior by guarding one-time picture-size configuration.
- Fixed potential stale capture gating by using ref-backed readiness checks during frame capture.
- Fixed YOLO tensor-layout detection branch so non-transposed output can be handled correctly.
- Removed commented debug code blocks and temporary console.log statements in recently modified runtime files.

### Documentation
- Updated Technical Appendix to current architecture and runtime behavior.
- Added this changelog for sprint-to-release traceability.

## [1.0.4] - 2026-04-03

### Changed
- Introduced ActiveCamera unified mode routing and global intent toggle.
- Added image URI decode path for optimized camera frame processing.
