# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and release versions follow [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

## [0.1.0] - 2026-08-03

This is the first stable release of Aspire. See the prerelease entries below
for the development history leading to this release.

### Changed

- Use `orng.RandomGenerator` for random number generation ([#83]).

### Fixed

- Report the installed package version correctly through `aspire.__version__`
  ([#84]).

## [0.1.0a21] - 2026-07-13

### Added

- Add a `stream` option to `configure_logger` ([#81]).

### Fixed

- Remove `n_steps` from keyword arguments after routing it to the appropriate
  sampler configuration ([#82]).

## [0.1.0a20] - 2026-05-26

### Added

- Add early stopping when training PyTorch flows ([#78]).
- Warn when the adaptive SMC inverse temperature does not change ([#76]).

### Changed

- Improve conversion between supported array namespaces ([#77]).
- Make sample statistics robust to NaN and infinite values ([#75]).
- Use `array-api-extra` for indexed array updates ([#55]).

## [0.1.0a19] - 2026-03-25

### Added

- Add broader support for MCMC samplers ([#71]).
- Add improved checkpoint creation, restoration, and configuration handling
  ([#51]).

### Changed

- Require Python 3.11 or later ([#74]).

### Fixed

- Preserve the requested dtype when sampling from a flow ([#67]).
- Pass missing sampling keyword arguments to their intended destination
  ([#68]).

## [0.1.0a18.post1] - 2026-03-02

### Fixed

- Fetch Git tags in the publishing workflow so `setuptools-scm` can determine
  the release version correctly ([#66]).

## [0.1.0a18] - 2026-03-02

### Added

- Add a maximum inverse-temperature step option to adaptive SMC ([#63]).

### Fixed

- Apply the maximum inverse-temperature step correctly ([#65]).
- Reject unsupported MiniPCN keyword arguments instead of silently ignoring
  them ([#64]).

## [0.1.0a17] - 2026-02-25

### Changed

- Suppress unwanted intermediate figures when plotting comparisons ([#61]).

### Fixed

- Only configure checkpointing for samplers that support it ([#62]).

## [0.1.0a16] - 2026-02-16

### Added

- Add an examples workflow to continuous integration ([#54]).
- Add checkpointing and result-saving examples ([#52]).
- Add iteration information to sample-history plots ([#58]).
- Enable SciPy Array API support automatically ([#56]).

### Fixed

- Correct dictionary conversion for `Samples` subclasses ([#57]).
- Correct `Samples.to_dataframe` output ([#60]).

## [0.1.0a15] - 2026-02-11

### Added

- Add methods for loading saved samples and histories ([#53]).

## [0.1.0a14] - 2026-02-10

### Added

- Add file-based logging ([#48]).
- Add richer SMC history information ([#45]).

### Changed

- Use PCN rather than TPCN in the default test configuration ([#50]).

### Fixed

- Correct the documented arguments for `Flow.sample` ([#49]).

## [0.1.0a13] - 2026-02-05

### Added

- Add a tolerance for adaptive inverse-temperature selection ([#42]).

### Changed

- Improve initial sample generation ([#41]).
- Correct dependency installation in continuous integration ([#43]).

### Fixed

- Set parameter names when initializing samplers ([#40]).

## [0.1.0a12] - 2025-12-16

### Added

- Add stable function identifiers for checkpoint serialization ([#39]).

## [0.1.0a11] - 2025-12-16

### Added

- Add checkpoint support to the SMC sampler ([#38]).

## [0.1.0a10] - 2025-12-04

### Added

- Add Array API support to the MiniPCN integration ([#37]).
- Document backend installation and the `aspire-bilby` companion package
  ([#36]).

## [0.1.0a9] - 2025-12-01

### Added

- Add a BlackJAX SMC example and integration test ([#34]).
- Expand documentation for transforms and multiprocessing ([#33]).

### Fixed

- Preserve `affine_transform` when recreating a `CompositeTransform` ([#35]).

## [0.1.0a8] - 2025-11-26

### Added

- Add the initial Sphinx documentation site ([#31], [#32]).

### Changed

- Improve errors for unknown flow backends ([#30]).

### Fixed

- Correct handling of the tuple returned by `get_flow_wrapper` ([#28]).

## [0.1.0a7] - 2025-11-03

### Fixed

- Pass the figure argument correctly in `Samples.plot_corner` ([#27]).

## [0.1.0a6] - 2025-10-14

### Added

- Allow `Aspire` to be initialized with an existing flow ([#22]).
- Add flow serialization ([#21]).
- Add explicit dtype configuration ([#24]).
- Add an SMC example ([#26]).

### Fixed

- Preserve dtype information in composite transforms ([#25]).

## [0.1.0a5] - 2025-09-23

### Added

- Add pickle support for sample objects ([#19]).

## [0.1.0a4] - 2025-09-23

### Fixed

- Include `tqdm` in the PyTorch optional dependencies ([#18]).

## [0.1.0a3] - 2025-09-23

### Fixed

- Support flows without a data transform ([#17]).

## [0.1.0a2] - 2025-09-19

### Added

- Add BlackJAX sampler support ([#8]).
- Add plotting of selected parameter subsets ([#12]).
- Add loading of the flow checkpoint with the best validation loss ([#14]).

### Changed

- Rename the package to `aspire` ([#16]).
- Rework the sampler and transform APIs ([#7]).
- Improve the flow and SMC implementations ([#10], [#15]).
- Enable adaptive SMC by default ([#11]).

### Fixed

- Use the NumPy Array API namespace in the NumPy SMC sampler ([#9]).
- Configure package loggers consistently ([#13]).

## [0.1.0a1] - 2025-06-13

### Added

- Add the initial `Aspire` and `Samples` APIs.
- Add PyTorch and JAX normalizing-flow backends.
- Add importance, MCMC, MiniPCN, and adaptive SMC samplers.
- Add bounded, periodic, affine, composite, and flow-based transforms.
- Add Array API-compatible NumPy, PyTorch, and JAX handling.
- Add sample, diagnostic-history, evidence, and plotting utilities.
- Add HDF5 serialization and configuration persistence.
- Add multiprocessing support and likelihood-evaluation tracking ([#6]).
- Add extensible flow selection and sampling configuration.
- Add testing, linting, and PyPI publishing workflows.
- Add the initial usage example and README documentation.

[Unreleased]: https://github.com/mj-will/aspire/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mj-will/aspire/compare/v0.1.0a21...v0.1.0
[0.1.0a21]: https://github.com/mj-will/aspire/compare/v0.1.0a20...v0.1.0a21
[0.1.0a20]: https://github.com/mj-will/aspire/compare/v0.1.0a19...v0.1.0a20
[0.1.0a19]: https://github.com/mj-will/aspire/compare/v0.1.0a18.post1...v0.1.0a19
[0.1.0a18.post1]: https://github.com/mj-will/aspire/compare/v0.1.0a18...v0.1.0a18.post1
[0.1.0a18]: https://github.com/mj-will/aspire/compare/v0.1.0a17...v0.1.0a18
[0.1.0a17]: https://github.com/mj-will/aspire/compare/v0.1.0a16...v0.1.0a17
[0.1.0a16]: https://github.com/mj-will/aspire/compare/v0.1.0a15...v0.1.0a16
[0.1.0a15]: https://github.com/mj-will/aspire/compare/v0.1.0a14...v0.1.0a15
[0.1.0a14]: https://github.com/mj-will/aspire/compare/v0.1.0a13...v0.1.0a14
[0.1.0a13]: https://github.com/mj-will/aspire/compare/v0.1.0a12...v0.1.0a13
[0.1.0a12]: https://github.com/mj-will/aspire/compare/v0.1.0a11...v0.1.0a12
[0.1.0a11]: https://github.com/mj-will/aspire/compare/v0.1.0a10...v0.1.0a11
[0.1.0a10]: https://github.com/mj-will/aspire/compare/v0.1.0a9...v0.1.0a10
[0.1.0a9]: https://github.com/mj-will/aspire/compare/v0.1.0a8...v0.1.0a9
[0.1.0a8]: https://github.com/mj-will/aspire/compare/v0.1.0a7...v0.1.0a8
[0.1.0a7]: https://github.com/mj-will/aspire/compare/v0.1.0a6...v0.1.0a7
[0.1.0a6]: https://github.com/mj-will/aspire/compare/v0.1.0a5...v0.1.0a6
[0.1.0a5]: https://github.com/mj-will/aspire/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/mj-will/aspire/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/mj-will/aspire/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/mj-will/aspire/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/mj-will/aspire/releases/tag/v0.1.0a1
[#6]: https://github.com/mj-will/aspire/pull/6
[#7]: https://github.com/mj-will/aspire/pull/7
[#8]: https://github.com/mj-will/aspire/pull/8
[#9]: https://github.com/mj-will/aspire/pull/9
[#10]: https://github.com/mj-will/aspire/pull/10
[#11]: https://github.com/mj-will/aspire/pull/11
[#12]: https://github.com/mj-will/aspire/pull/12
[#13]: https://github.com/mj-will/aspire/pull/13
[#14]: https://github.com/mj-will/aspire/pull/14
[#15]: https://github.com/mj-will/aspire/pull/15
[#16]: https://github.com/mj-will/aspire/pull/16
[#17]: https://github.com/mj-will/aspire/pull/17
[#18]: https://github.com/mj-will/aspire/pull/18
[#19]: https://github.com/mj-will/aspire/pull/19
[#21]: https://github.com/mj-will/aspire/pull/21
[#22]: https://github.com/mj-will/aspire/pull/22
[#24]: https://github.com/mj-will/aspire/pull/24
[#25]: https://github.com/mj-will/aspire/pull/25
[#26]: https://github.com/mj-will/aspire/pull/26
[#27]: https://github.com/mj-will/aspire/pull/27
[#28]: https://github.com/mj-will/aspire/pull/28
[#30]: https://github.com/mj-will/aspire/pull/30
[#31]: https://github.com/mj-will/aspire/pull/31
[#32]: https://github.com/mj-will/aspire/pull/32
[#33]: https://github.com/mj-will/aspire/pull/33
[#34]: https://github.com/mj-will/aspire/pull/34
[#35]: https://github.com/mj-will/aspire/pull/35
[#36]: https://github.com/mj-will/aspire/pull/36
[#37]: https://github.com/mj-will/aspire/pull/37
[#38]: https://github.com/mj-will/aspire/pull/38
[#39]: https://github.com/mj-will/aspire/pull/39
[#40]: https://github.com/mj-will/aspire/pull/40
[#41]: https://github.com/mj-will/aspire/pull/41
[#42]: https://github.com/mj-will/aspire/pull/42
[#43]: https://github.com/mj-will/aspire/pull/43
[#45]: https://github.com/mj-will/aspire/pull/45
[#48]: https://github.com/mj-will/aspire/pull/48
[#49]: https://github.com/mj-will/aspire/pull/49
[#50]: https://github.com/mj-will/aspire/pull/50
[#51]: https://github.com/mj-will/aspire/pull/51
[#52]: https://github.com/mj-will/aspire/pull/52
[#53]: https://github.com/mj-will/aspire/pull/53
[#54]: https://github.com/mj-will/aspire/pull/54
[#55]: https://github.com/mj-will/aspire/pull/55
[#56]: https://github.com/mj-will/aspire/pull/56
[#57]: https://github.com/mj-will/aspire/pull/57
[#58]: https://github.com/mj-will/aspire/pull/58
[#60]: https://github.com/mj-will/aspire/pull/60
[#61]: https://github.com/mj-will/aspire/pull/61
[#62]: https://github.com/mj-will/aspire/pull/62
[#63]: https://github.com/mj-will/aspire/pull/63
[#64]: https://github.com/mj-will/aspire/pull/64
[#65]: https://github.com/mj-will/aspire/pull/65
[#66]: https://github.com/mj-will/aspire/pull/66
[#67]: https://github.com/mj-will/aspire/pull/67
[#68]: https://github.com/mj-will/aspire/pull/68
[#71]: https://github.com/mj-will/aspire/pull/71
[#74]: https://github.com/mj-will/aspire/pull/74
[#75]: https://github.com/mj-will/aspire/pull/75
[#76]: https://github.com/mj-will/aspire/pull/76
[#77]: https://github.com/mj-will/aspire/pull/77
[#78]: https://github.com/mj-will/aspire/pull/78
[#81]: https://github.com/mj-will/aspire/pull/81
[#82]: https://github.com/mj-will/aspire/pull/82
[#83]: https://github.com/mj-will/aspire/pull/83
[#84]: https://github.com/mj-will/aspire/pull/84
