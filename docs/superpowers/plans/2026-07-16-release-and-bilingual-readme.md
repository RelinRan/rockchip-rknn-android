# Release and Bilingual README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the library independently buildable and publish a versioned AAR with bilingual documentation and an MIT license.

**Architecture:** Gradle project metadata supplies a single version source. A tag-driven GitHub Actions job validates that version, tests and builds the library, then publishes the AAR and checksum.

**Tech Stack:** Gradle Kotlin DSL, Android Gradle Plugin, Kotlin, GitHub Actions

---

### Task 1: Standalone build

**Files:** `settings.gradle.kts`, `gradle.properties`, `gradle/libs.versions.toml`, `build.gradle.kts`, Gradle wrapper files

- [ ] Add plugin/dependency repositories and the version catalog.
- [ ] Set `VERSION_NAME=1.0.0` as the project version.
- [ ] Generate the Gradle wrapper and run `testDebugUnitTest assembleRelease`.

### Task 2: Documentation and license

**Files:** `README.md`, `README.zh-CN.md`, `LICENSE`

- [ ] Make English the default README and add reciprocal language links.
- [ ] Document official YOLO and RKNN-Toolkit2 relationships.
- [ ] Add the complete MIT license text.

### Task 3: Automated release

**Files:** `.github/workflows/release.yml`

- [ ] Trigger on semantic `vX.Y.Z` tags and allow manual diagnostics.
- [ ] Validate the tag against `VERSION_NAME`, run tests, and build the release AAR with detailed logs.
- [ ] Upload a versioned AAR and SHA-256 file to workflow artifacts and GitHub Releases.
