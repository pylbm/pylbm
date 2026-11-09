# Changelog

## [0.12.0](https://github.com/pylbm/pylbm/compare/v0.11.0...v0.12.0) (2026-11-09)


### Features

* add a source to the distribution functions after the relaxation ([#196](https://github.com/pylbm/pylbm/issues/196)) ([eb57ee1](https://github.com/pylbm/pylbm/commit/eb57ee125157e3b9b2bcccb63c4a04d9f4df1af9))
* **boundary:** add interpolated bounce-back conditions with a post-collision correction ([#204](https://github.com/pylbm/pylbm/issues/204)) ([48c19da](https://github.com/pylbm/pylbm/commit/48c19da71cff3e9d7ebd20309aea64d27c138b06))
* **boundary:** choose the boundary condition by velocity and warn when a link reads a solid node ([#206](https://github.com/pylbm/pylbm/issues/206)) ([4d63b6c](https://github.com/pylbm/pylbm/commit/4d63b6ce37c5bb25dd2aac03fa46f2d37796419c))
* **boundary:** rename the Bouzidi conditions to BFLBounceBack and BFLAntiBounceBack ([#205](https://github.com/pylbm/pylbm/issues/205)) ([12f05a4](https://github.com/pylbm/pylbm/commit/12f05a428a3f75c156f5405efbf23504354bebb8))
* define the conserved moments with their own polynomials ([#195](https://github.com/pylbm/pylbm/issues/195)) ([ba7adb9](https://github.com/pylbm/pylbm/commit/ba7adb94297ec90d9fe3b0667c8524d0df978643))
* run the GPU backend on loopy 2025.2 instead of loo.py 2017.2 ([#190](https://github.com/pylbm/pylbm/issues/190)) ([4ef7336](https://github.com/pylbm/pylbm/commit/4ef73368ba141cfe18fc39e3ca4801c54492c192))


### Bug Fixes

* correct the Bouzidi conditions off half a link and the wall corners next to a periodic face ([#192](https://github.com/pylbm/pylbm/issues/192)) ([5c4780a](https://github.com/pylbm/pylbm/commit/5c4780af0803d3103553fdfcb046821fc9499aee))
* generate the same Loopy code at each run ([#200](https://github.com/pylbm/pylbm/issues/200)) ([9a7e484](https://github.com/pylbm/pylbm/commit/9a7e4846b7fb9bbd25feacd3fdc7a407f48e4040))
* give the same BouzidiAntiBounceBack results in every backend near curved walls ([#203](https://github.com/pylbm/pylbm/issues/203)) ([ef834e7](https://github.com/pylbm/pylbm/commit/ef834e7fbbd82c2e7a88995d5d2d824357f62de6))
* give the space variables to the Loopy kernels as device arrays ([#197](https://github.com/pylbm/pylbm/issues/197)) ([0088da3](https://github.com/pylbm/pylbm/commit/0088da309d637dfa20d4583b009eb3bd74559ddd))
* make the check_isfluid setting work in every backend ([#201](https://github.com/pylbm/pylbm/issues/201)) ([6d3704b](https://github.com/pylbm/pylbm/commit/6d3704be401cc5ea959991d57a8317985a5754fd))
* make the split and m_local settings work in every backend ([#199](https://github.com/pylbm/pylbm/issues/199)) ([484181c](https://github.com/pylbm/pylbm/commit/484181c3d56cc5fa38cc8f80f9f69e00286533e1))
* point the last loopy ImportError message at pylbm[gpu] ([#194](https://github.com/pylbm/pylbm/issues/194)) ([850bc2c](https://github.com/pylbm/pylbm/commit/850bc2c87103cd832c791dd64bd079abc1d610fa))
* run on Python 3.13+ and fix space-dependent schemes and storage orders ([#191](https://github.com/pylbm/pylbm/issues/191)) ([30c81c5](https://github.com/pylbm/pylbm/commit/30c81c55f4ebee3fd0812062acc6ee856bb51ffc))
* run the bounce-back conditions and several simulations with Loopy ([#198](https://github.com/pylbm/pylbm/issues/198)) ([842813c](https://github.com/pylbm/pylbm/commit/842813c25d61a44c3676d845eb80dfdb2b40cd91))
* unblock CI on pytest 9, matplotlib 3.11, GCC 15 and the new runners ([#193](https://github.com/pylbm/pylbm/issues/193)) ([7b8ea56](https://github.com/pylbm/pylbm/commit/7b8ea5630ea2157cfc50b3cd12489a6135f631a7))

## [0.11.0](https://github.com/pylbm/pylbm/compare/0.10.0...v0.11.0) (2026-05-14)


### Features

* add release please ([#187](https://github.com/pylbm/pylbm/issues/187)) ([2230e93](https://github.com/pylbm/pylbm/commit/2230e93683b83ba3f73e3f9e47534adca8887e12))


### Bug Fixes

* remove release-please on push in master ([#189](https://github.com/pylbm/pylbm/issues/189)) ([9818aaa](https://github.com/pylbm/pylbm/commit/9818aaaa9ee2864b49cd0da893a9ecdd608ec275))
