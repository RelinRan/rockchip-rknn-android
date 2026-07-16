# Release and Bilingual README Design

The repository will be independently buildable as a single Android library. `README.md` is the English default, `README.zh-CN.md` preserves the complete Chinese documentation, and reciprocal language links appear at the top of both files.

Project version `1.0.0` is stored once in `gradle.properties`. A `v1.0.0`-style tag triggers tests, a release AAR build, artifact inspection and hashing, and GitHub Release publication. The workflow rejects tags that disagree with the configured version and retains manual dispatch for diagnostic builds without publishing a release.

The project is licensed under MIT and links to the official Ultralytics YOLO and RKNN-Toolkit2 Android Runtime API resources.
