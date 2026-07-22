# Changelog

## [1.1.0](https://github.com/NFDI4Chem/nmrkit/compare/v1.0.0...v1.1.0) (2026-07-22)


### Features

* improve prediction with molfile ([ab20577](https://github.com/NFDI4Chem/nmrkit/commit/ab20577306584b3d0cf7f5cec64dc70125a150b5))


### Bug Fixes

* add scalar-fastapi to requirements.txt for scalar API support ([#119](https://github.com/NFDI4Chem/nmrkit/issues/119)) ([965abd7](https://github.com/NFDI4Chem/nmrkit/commit/965abd71c719911e77672100bd97bd12eba99543))
* numpy version to &lt;2 to resolve compatibility issues ([#117](https://github.com/NFDI4Chem/nmrkit/issues/117)) ([4c5881a](https://github.com/NFDI4Chem/nmrkit/commit/4c5881aa1ff2a518c1f56674d90114d0ff513a3c))


### Documentation

* ignore dead links in localhost for vitepress config ([e78d7cf](https://github.com/NFDI4Chem/nmrkit/commit/e78d7cfbd83707d567ac66012f02c28bdde6ecf3))
* migrate to Scalar and complete documentation update ([#114](https://github.com/NFDI4Chem/nmrkit/issues/114)) ([34b2195](https://github.com/NFDI4Chem/nmrkit/commit/34b2195718d0e68d813b354954ed932b7c89c898))

## [0.2.0](https://github.com/NFDI4Chem/nmrkit/compare/v0.1.0...v0.2.0) (2026-07-15)


### Features

* add automated build for nmr-cli ([175a1ec](https://github.com/NFDI4Chem/nmrkit/commit/175a1ec1388396495e261d7cf849e6e6917b7a63))
* add automated build for nmr-cli ([27d568a](https://github.com/NFDI4Chem/nmrkit/commit/27d568a46e713cd68d25304f80ac12413c6e6c63))
* add docker-compose file for dev and prod ([631eefc](https://github.com/NFDI4Chem/nmrkit/commit/631eefcc9495432b0021c2925a7b9aeb9d1dd54d))
* add more test for chem ([564eec6](https://github.com/NFDI4Chem/nmrkit/commit/564eec630870a7adf9a1c284f439eacd335babe6))
* add spectra parsing API for file upload and URL input ([35ce1ad](https://github.com/NFDI4Chem/nmrkit/commit/35ce1ad840492191bc99629bd23ea56256adcb14))
* add spectra parsing API for file upload and URL input ([712d5dd](https://github.com/NFDI4Chem/nmrkit/commit/712d5ddaba9aa1c955858607820e4e814ac8baf2))
* add support for selecting peak shape algorithm (lorentzian or gaussian) ([67c17ac](https://github.com/NFDI4Chem/nmrkit/commit/67c17ac46faf69918d4bdce94e92818785713a68))
* auto processing 1d core functions ([481441c](https://github.com/NFDI4Chem/nmrkit/commit/481441ce3278d077b944453705d1f3c9225a7946))
* auto processing 2d core functions ([a445d85](https://github.com/NFDI4Chem/nmrkit/commit/a445d85a59678165617d1fb0fb3e51b3809e60f7))
* auto ranges detection core function ([d152538](https://github.com/NFDI4Chem/nmrkit/commit/d152538c34e192995b0492879b6e35e27e56dcc4))
* auto zones detection core function ([376d8c3](https://github.com/NFDI4Chem/nmrkit/commit/376d8c3cf9542e707f77bb5bc5e9d4cd8e3ab9fc))
* expose nmrium features ([8f125be](https://github.com/NFDI4Chem/nmrkit/commit/8f125beffed519eb34e076620d2cb06c445e374d))
* expose options for automatic processing and range/zone detection ([2d84e7e](https://github.com/NFDI4Chem/nmrkit/commit/2d84e7e9e3b16eebcecffbfad61e2f1e6dba5927))
* improve prediction with molfile ([3cfb355](https://github.com/NFDI4Chem/nmrkit/commit/3cfb3551c424060341f283d70c01678c6318e8b5))
* integrate nmr-respredict docker image ([11fdebb](https://github.com/NFDI4Chem/nmrkit/commit/11fdebba26f7ef5dc517749234a7e2edd65a456e))
* integrated respredict ([66ad270](https://github.com/NFDI4Chem/nmrkit/commit/66ad27072fb3a04bcacf7f5f3911fbc9f40bd6c8))
* migrate to nmrium-core and nmrium-core-plugins ([3e1cc62](https://github.com/NFDI4Chem/nmrkit/commit/3e1cc624b81e99945d4817d9535a083e6ef9e8d5))
* **nmr-cli:** update NMRium core packages ([b9b2cbb](https://github.com/NFDI4Chem/nmrkit/commit/b9b2cbb5c40c6a9a135cd6b7c11bdfc9c01ee39c))
* nmrium snapshot cli ([3b39e8e](https://github.com/NFDI4Chem/nmrkit/commit/3b39e8ed29c306024f989444754a044de37da04a))
* predict spectrum from mol using NMRShiftDB ([8e981d0](https://github.com/NFDI4Chem/nmrkit/commit/8e981d042758e9fe698ef8159b8051fe74e484c0))
* predict spectrum from mol using NMRShiftDB ([d510131](https://github.com/NFDI4Chem/nmrkit/commit/d5101310f0d160de4d6f1220898fe555f1075f70))
* resurrect spectrum from publication string ([d90c2fc](https://github.com/NFDI4Chem/nmrkit/commit/d90c2fc2e89996f955ede2471622dce9ff8fd0a6))
* resurrect spectrum from publication string ([375c62f](https://github.com/NFDI4Chem/nmrkit/commit/375c62ff8b4ce99c3fb8ad8524d673ec751b00b2))
* update docker image to playwright:v1.43.1-jammy ([913a4d5](https://github.com/NFDI4Chem/nmrkit/commit/913a4d5531245e9dfd7c49f1706be869cc3dc063))
* update nmr-load-save to version  2.1.0 ([9e40ba8](https://github.com/NFDI4Chem/nmrkit/commit/9e40ba84b117beaafcbd3aa575cf5af7311f9ce8))
* update nmr-load-save to version  3.1.3 ([1ab07e5](https://github.com/NFDI4Chem/nmrkit/commit/1ab07e5507ff0dd96ded0f1666fa0ad80b6c4d23))
* update nmr-load-save to version 0.23.11 ([ef91afc](https://github.com/NFDI4Chem/nmrkit/commit/ef91afcce3a6ff3b1e6de212f99c3cdfa1453de8))
* update nmr-load-save to version 0.28.0 ([#75](https://github.com/NFDI4Chem/nmrkit/issues/75)) ([99f6986](https://github.com/NFDI4Chem/nmrkit/commit/99f6986066b3c0723dd1d2740d8dc7b7cb003581))
* update nmr-load-save to version 0.29.3 ([bed41c1](https://github.com/NFDI4Chem/nmrkit/commit/bed41c14dd45897152eb17ad329474a07f7a87dc))
* update nmr-load-save to version 2.1.0 ([f1df45d](https://github.com/NFDI4Chem/nmrkit/commit/f1df45db5994003c03c26aacab1dfed947d6cfe0))
* update nmr-load-save to version 3.1.3 ([2ce03a0](https://github.com/NFDI4Chem/nmrkit/commit/2ce03a0f4bbd7e134d8f0e693d383c3282ac8cb7))
* update nmr-load-save to version 3.3.0 ([2be8067](https://github.com/NFDI4Chem/nmrkit/commit/2be806730d9f5282f99453338caaf20aa4c2b40a))
* update nmr-load-save to version 3.6.0 ([#82](https://github.com/NFDI4Chem/nmrkit/issues/82)) ([6e352d5](https://github.com/NFDI4Chem/nmrkit/commit/6e352d5318f1f5d738ab558c654a3594efd66d57))
* update NMRium core packages ([423da54](https://github.com/NFDI4Chem/nmrkit/commit/423da54cc07966ba74ccd32801a50f324340d41d))
* update NMRium core packages ([806dae0](https://github.com/NFDI4Chem/nmrkit/commit/806dae0d4c9a974c74b60b437b5a0da85ddd7d34))
* update nmrium packages to latest version ([5faa671](https://github.com/NFDI4Chem/nmrkit/commit/5faa671c08b07296d4dac94d69567545a02bdac6))
* update to nmr-load-save version 0.33.1 ([eecf2e5](https://github.com/NFDI4Chem/nmrkit/commit/eecf2e5df40b7639c68584bb8fe7b1031639070a))
* update workflow files to use test and lint code ([bae4d4a](https://github.com/NFDI4Chem/nmrkit/commit/bae4d4afc8ad61939bc4f91efc88195d07163f17))
* updated health check endpoints ([a08b2e2](https://github.com/NFDI4Chem/nmrkit/commit/a08b2e2b5ddc13ec10e78b8549600b23b8c6830e))


### Bug Fixes

* add docker to nmr-cli image ([01c0e21](https://github.com/NFDI4Chem/nmrkit/commit/01c0e21b52a641472cfecbe419e26aa3501b72da))
* add docker to nmr-cli image ([4628360](https://github.com/NFDI4Chem/nmrkit/commit/4628360cb0a1e21e32868f08886bd408c6557aa7))
* add minio creds to .env.template ([fb65894](https://github.com/NFDI4Chem/nmrkit/commit/fb658940fb0d8d9aec189de8e90701830ad45b7e))
* add nmr-snapshot to docker-compose ([389903d](https://github.com/NFDI4Chem/nmrkit/commit/389903dc50dd4c0fbf4e954359c5e35e1fb1616a))
* add Orcid Id for Hamed ([d861f1b](https://github.com/NFDI4Chem/nmrkit/commit/d861f1b85547614c4ddfb51691da4c0c4c65080b))
* added lwreg to flake8 ([92aa647](https://github.com/NFDI4Chem/nmrkit/commit/92aa64703551fade50f9bd5208f7dd91e6da0e33))
* added nmr-respredict to flake8 exclusion list ([8e4392b](https://github.com/NFDI4Chem/nmrkit/commit/8e4392bc1f93b33b4e9581df43315c7115c59fcf))
* added spectra router ([62f6981](https://github.com/NFDI4Chem/nmrkit/commit/62f6981469b85397b298c979326c455a814a42d4))
* ASGI app loading issue fix ([58b1ea2](https://github.com/NFDI4Chem/nmrkit/commit/58b1ea2e42b169acb7b30e4a60f42e0d5334ac0f))
* docker build issues fix ([35b6523](https://github.com/NFDI4Chem/nmrkit/commit/35b652379a78405ab7d4e6b985f1c3db38b3a85a))
* enabled cli and respredict on the dev docker compose file ([433542a](https://github.com/NFDI4Chem/nmrkit/commit/433542af9fab14655f94b9e8065af91d028ee9a0))
* flake8 errors fix ([927cc1c](https://github.com/NFDI4Chem/nmrkit/commit/927cc1c9baeb833bc9e34caf34271026ea33bd25))
* ignore lwreg in pylint ([4707d98](https://github.com/NFDI4Chem/nmrkit/commit/4707d9806036957845e6fe5b6f9441904c5ea58f))
* ignore lwreg test cases ([23cd228](https://github.com/NFDI4Chem/nmrkit/commit/23cd22853f08ab12f50da756dcb8c3ac907bcefa))
* implemented subprocess to exec nmr-cli on converter image ([85f1ea3](https://github.com/NFDI4Chem/nmrkit/commit/85f1ea35dee7e3e08e665b4d2d3fda540432c186))
* install flake8 in release-please ([2388fb9](https://github.com/NFDI4Chem/nmrkit/commit/2388fb97e0fe796e1dbb29d432628845e9ff04a4))
* install flake8 in release-please ([45ff613](https://github.com/NFDI4Chem/nmrkit/commit/45ff6134a6c83ea2663f4daf5a557bd56559685d))
* label-atom test and typos ([e0bded8](https://github.com/NFDI4Chem/nmrkit/commit/e0bded8ee8641d95aa5cc157031d1e6c89defd84))
* label-atom test and typos ([bd294fd](https://github.com/NFDI4Chem/nmrkit/commit/bd294fdf711da6d2d6486c4d736950f564799e0a))
* more docker file build issue fixes ([b97feb8](https://github.com/NFDI4Chem/nmrkit/commit/b97feb8a1ccc9178f8753f797de0e78e1e950395))
* mount docker.sock ([d3ee617](https://github.com/NFDI4Chem/nmrkit/commit/d3ee617f282b8cd05f084966429d4554dd60ff47))
* mount docker.sock ([d9178cd](https://github.com/NFDI4Chem/nmrkit/commit/d9178cd8a83e0cb3941318b382f927145b0741e8))
* move dockerhub to nfdi4chem namespace ([b25ada4](https://github.com/NFDI4Chem/nmrkit/commit/b25ada4656723314d752a5df1c290ae136241b76))
* moved HOSE code endpoint to chem router and various other reorganisational changes ([41f50b7](https://github.com/NFDI4Chem/nmrkit/commit/41f50b7b86e0e57bc0640f4eae761c1fc5267766))
* pull image from Docker Hub instead of ([f7f8e42](https://github.com/NFDI4Chem/nmrkit/commit/f7f8e427e01ae27d177b3a261565007bee366c37))
* remove unused parameter from .env template ([93cc891](https://github.com/NFDI4Chem/nmrkit/commit/93cc891833a2491e9bae3008e1155d425de3ebad))
* test failures ([3cf6077](https://github.com/NFDI4Chem/nmrkit/commit/3cf6077b55f22ebf42885080d7ef311bf39e74bc))
* test fucntions ([f02fa6e](https://github.com/NFDI4Chem/nmrkit/commit/f02fa6e43279bd843e7a2260d42561aad747b193))
* test functions ([b4437eb](https://github.com/NFDI4Chem/nmrkit/commit/b4437ebabb71bdb62dfca97064f5bb5a5c81c780))
* update .env.template ([fe3bc37](https://github.com/NFDI4Chem/nmrkit/commit/fe3bc37e91a3eb1ea38df8f92e25a0ce3dad122e))
* update Citation.cff ([8695c2f](https://github.com/NFDI4Chem/nmrkit/commit/8695c2f2ce0f4c051685dd4b5ac0618f3d9c52ae))
* update Citation.cff ([cd83a27](https://github.com/NFDI4Chem/nmrkit/commit/cd83a2701c4fb6e42bcd96c3d9bdf2ec768fc0dd))
* update Citation.cff ([#48](https://github.com/NFDI4Chem/nmrkit/issues/48)) ([0e01592](https://github.com/NFDI4Chem/nmrkit/commit/0e01592d7904b720a0e27269610f13e0cc61bfac))
* update converter api decription ([56d2bfd](https://github.com/NFDI4Chem/nmrkit/commit/56d2bfda13d3ed45816914c46065d8f2994af884))
* update docker compose to add nmr-respredict ([69bfdff](https://github.com/NFDI4Chem/nmrkit/commit/69bfdffe3f7264bde87c318f7aef9f0e525d2f84))
* update ignored path for lwreg ([24553f3](https://github.com/NFDI4Chem/nmrkit/commit/24553f38e736d03d8f91e3092cbe4bc4c780ec85))
* update job name grafana dashboard json ([d9d89f8](https://github.com/NFDI4Chem/nmrkit/commit/d9d89f8548c8a1442603183383c6c9d02c9e4304))
* update job name grafana dashboard json ([78152c0](https://github.com/NFDI4Chem/nmrkit/commit/78152c0e154bf8212870f02b287df0981a34a60e))
* update workflow files to use test ([68ed8d6](https://github.com/NFDI4Chem/nmrkit/commit/68ed8d66a4aeab2fcf15ffbfaa0fc3e9e2122928))
* workflow updates ([0a348c3](https://github.com/NFDI4Chem/nmrkit/commit/0a348c3daf486c58d538c0c275196d4bb0068fed))


### Documentation

* add DFG credit in READMe ([386ae23](https://github.com/NFDI4Chem/nmrkit/commit/386ae23338bab6bbb84c9d9694ac167111da9aab))
* add DFG credit in READMe ([5c3f450](https://github.com/NFDI4Chem/nmrkit/commit/5c3f4509a4f12e4889a7e08f1422499dbf09b021))
* remove danger alert ([2b538b8](https://github.com/NFDI4Chem/nmrkit/commit/2b538b85a3d851359f7f4d19deca378c17154983))
* remove danger alert ([06c88e1](https://github.com/NFDI4Chem/nmrkit/commit/06c88e18fdb565b9fae1ea5760df23be5c70aa5d))
* update API endpoint ([2b4250e](https://github.com/NFDI4Chem/nmrkit/commit/2b4250e187e04555117f3efa89bdc5a2c5084247))
* update API endpoint ([5625b41](https://github.com/NFDI4Chem/nmrkit/commit/5625b41c9b4cd32827e57c862e7125b6ba919eee))
* update license ([474e407](https://github.com/NFDI4Chem/nmrkit/commit/474e407ed2768d99f2eb321171b5f45232cea133))
* update license ([db12f8c](https://github.com/NFDI4Chem/nmrkit/commit/db12f8c60b14df1f164a5fedc6e62ca88f21370e))
* update pages-docker,cluster-deployment, ([78e3d72](https://github.com/NFDI4Chem/nmrkit/commit/78e3d72c7a463b7bf351405f5d3a45887589fa76))
* update pages-docker,cluster-deployment, ([ce67ac1](https://github.com/NFDI4Chem/nmrkit/commit/ce67ac13a505c3434aefe13950827be6df758597))
* update README ([ced1c62](https://github.com/NFDI4Chem/nmrkit/commit/ced1c62f0e9802997a305fc0796cb20947209945))
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
