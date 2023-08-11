# Changelog

## [0.2.0](https://github.com/NFDI4Chem/nmrkit/compare/v0.1.0...v0.2.0) (2023-08-11)


### Features

* add docker-compose file for dev and prod ([631eefc](https://github.com/NFDI4Chem/nmrkit/commit/631eefcc9495432b0021c2925a7b9aeb9d1dd54d))
* updated health check endpoints ([a08b2e2](https://github.com/NFDI4Chem/nmrkit/commit/a08b2e2b5ddc13ec10e78b8549600b23b8c6830e))


### Bug Fixes

* add minio creds to .env.template ([fb65894](https://github.com/NFDI4Chem/nmrkit/commit/fb658940fb0d8d9aec189de8e90701830ad45b7e))
* add Orcid Id for Hamed ([d861f1b](https://github.com/NFDI4Chem/nmrkit/commit/d861f1b85547614c4ddfb51691da4c0c4c65080b))
* added spectra router ([62f6981](https://github.com/NFDI4Chem/nmrkit/commit/62f6981469b85397b298c979326c455a814a42d4))
* flake8 errors fix ([927cc1c](https://github.com/NFDI4Chem/nmrkit/commit/927cc1c9baeb833bc9e34caf34271026ea33bd25))
* install flake8 in release-please ([45ff613](https://github.com/NFDI4Chem/nmrkit/commit/45ff6134a6c83ea2663f4daf5a557bd56559685d))
* move dockerhub to nfdi4chem namespace ([b25ada4](https://github.com/NFDI4Chem/nmrkit/commit/b25ada4656723314d752a5df1c290ae136241b76))
* moved HOSE code endpoint to chem router and various other reorganisational changes ([41f50b7](https://github.com/NFDI4Chem/nmrkit/commit/41f50b7b86e0e57bc0640f4eae761c1fc5267766))
* remove unused parameter from .env template ([93cc891](https://github.com/NFDI4Chem/nmrkit/commit/93cc891833a2491e9bae3008e1155d425de3ebad))
* update Citation.cff ([8695c2f](https://github.com/NFDI4Chem/nmrkit/commit/8695c2f2ce0f4c051685dd4b5ac0618f3d9c52ae))
* update Citation.cff ([cd83a27](https://github.com/NFDI4Chem/nmrkit/commit/cd83a2701c4fb6e42bcd96c3d9bdf2ec768fc0dd))
* update Citation.cff ([#48](https://github.com/NFDI4Chem/nmrkit/issues/48)) ([0e01592](https://github.com/NFDI4Chem/nmrkit/commit/0e01592d7904b720a0e27269610f13e0cc61bfac))
* update job name grafana dashboard json ([78152c0](https://github.com/NFDI4Chem/nmrkit/commit/78152c0e154bf8212870f02b287df0981a34a60e))


### Documentation

* add DFG credit in READMe ([5c3f450](https://github.com/NFDI4Chem/nmrkit/commit/5c3f4509a4f12e4889a7e08f1422499dbf09b021))
* remove danger alert ([06c88e1](https://github.com/NFDI4Chem/nmrkit/commit/06c88e18fdb565b9fae1ea5760df23be5c70aa5d))
* update API endpoint ([5625b41](https://github.com/NFDI4Chem/nmrkit/commit/5625b41c9b4cd32827e57c862e7125b6ba919eee))
* update license ([db12f8c](https://github.com/NFDI4Chem/nmrkit/commit/db12f8c60b14df1f164a5fedc6e62ca88f21370e))
* update pages-docker,cluster-deployment, ([ce67ac1](https://github.com/NFDI4Chem/nmrkit/commit/ce67ac13a505c3434aefe13950827be6df758597))
* update README ([5e6a3a6](https://github.com/NFDI4Chem/nmrkit/commit/5e6a3a6d095f3e09e58d92487a463f6ae184db84))

## 0.1.0 (2023-08-03)


### Features

* add release-please ([36e5efb](https://github.com/NFDI4Chem/nmrkit/commit/36e5efbe0162d6a8722b97b71381d7e0c6ad8f0b))
* added docs deploy actions ([7ba281e](https://github.com/NFDI4Chem/nmrkit/commit/7ba281ec0a6f3ea377c07894300c189fcb1a1a62))
* bootstrap vitepress documentation ([65d0a36](https://github.com/NFDI4Chem/nmrkit/commit/65d0a366f060cb7a2f7d7bdb2563ed49bacb9346)), closes [#18](https://github.com/NFDI4Chem/nmrkit/issues/18)
* bootstrapped fastapi with rdkit & cdk ([b277e68](https://github.com/NFDI4Chem/nmrkit/commit/b277e68c43a6c201b6b289fee303b831af8bacd6))
* implemented prometheus and graphana logging, added gitignore, citation.cff, versioning to REST API ([0c55ea0](https://github.com/NFDI4Chem/nmrkit/commit/0c55ea0acc9da9f99408201f8814ff50123c9a51))
* updated chem router annotations and other formatting changes ([b4b4e45](https://github.com/NFDI4Chem/nmrkit/commit/b4b4e45565ae973279c12fd530172812577d7d34))


### Bug Fixes

* add grafana_data folder to avoid Error response from daemon: failed to mount local volume: mount ./grafana_data:/var/lib/docker/volumes/nmr-predict_grafana_data/_data, flags: 0x1000: no such file or directory issue ([352784a](https://github.com/NFDI4Chem/nmrkit/commit/352784ad882b7f27f322edc58c149aca6d555fe3))
* added base path ([4a68223](https://github.com/NFDI4Chem/nmrkit/commit/4a682234e4eb1d2be1b32e54fe8a2378c9447f26))
* added env template ([b5a09d4](https://github.com/NFDI4Chem/nmrkit/commit/b5a09d45a884950a07238ee82acec5d0bdda7226))
* added grafana_data folder ([fdbf8d0](https://github.com/NFDI4Chem/nmrkit/commit/fdbf8d0f6ae5ce87275c62da41daff71d506172d))
* default home page redirect issue fix ([55bc668](https://github.com/NFDI4Chem/nmrkit/commit/55bc6688b62704a61f6cd75afac32fc413c1fb8b))
* name and readme updates ([3a403d6](https://github.com/NFDI4Chem/nmrkit/commit/3a403d6105455edf173ad5ed92df050f42af3404))
* name changes for ms ([1cedba0](https://github.com/NFDI4Chem/nmrkit/commit/1cedba0f39d2c9a4f71924b2e8f5dae52226e81f))
* resolve dependency issues, added logo and other changes ([b968e7c](https://github.com/NFDI4Chem/nmrkit/commit/b968e7cb65e01adf1dd6b2648c80706d61420354))
* resolved volume name issues and added prometheus yaml file ([1ddef31](https://github.com/NFDI4Chem/nmrkit/commit/1ddef310f7e71c9d68c8d94f1fec52c5dc67c4e5))
* ResolvePackageNotFound: [#12](https://github.com/NFDI4Chem/nmrkit/issues/12) 110.6 - conda==23.5.2 issue fix ([aaf0e9b](https://github.com/NFDI4Chem/nmrkit/commit/aaf0e9b6075676a3c3dea3d36110ddef792a3f7b))
* updated logo ([c50884d](https://github.com/NFDI4Chem/nmrkit/commit/c50884dd0844d8356966bc557ca4d95274e5e849))
* Vite config title and description update ([edb1c0f](https://github.com/NFDI4Chem/nmrkit/commit/edb1c0f0e6a2ff551a2dc1f2a3e39dd1de91a53d))


### Documentation

* update nav ([d6d4a22](https://github.com/NFDI4Chem/nmrkit/commit/d6d4a222b516561570e40d1abc925300cbc4651a))
* update theme ([9d8b1a8](https://github.com/NFDI4Chem/nmrkit/commit/9d8b1a8d86b50dd80a8992ed293a7c6131e167e2))
