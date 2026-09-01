import { join, isAbsolute } from 'node:path';
import type { ParsingOptions, NmriumState } from '@zakodium/nmrium-core';
import init from '@zakodium/nmrium-core-plugins';
import { FileCollection } from 'file-collection';
import yargs from 'yargs';
import { FifoLogger } from 'fifo-logger';
import { FileOptionsArgs } from '..';
import { runSpectraPipeline } from './run-pipeline';
import { outputResult } from './utility/outputResult';
import { toMessage } from './utility/toMessage';

const core = init();


type RequiredKey<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>;


function getParsingOptions(autoProcessing: boolean): ParsingOptions {
  return {
    onLoadProcessing: { autoProcessing },
    selector: { general: { dataSelection: 'preferFT' } },
    experimentalFeatures: true,
  };
}

async function processAndSerialize(
  nmriumState: Partial<NmriumState>,
  options: FileOptionsArgs,
  logger: FifoLogger
) {
  const { s: enableSnapshot = false, p: autoProcessing = false, d: autoDetection = false, o, r } = options;

  const images = await runSpectraPipeline(nmriumState, { autoProcessing, autoDetection, enableSnapshot }, logger);

  const { data, version } = core.serializeNmriumState(
    nmriumState as NmriumState,
    { includeData: r ? 'rawData' : 'dataSource' },
  );

  // include the meta and info object in case of serialize as dataSource
  const spectra: any = data?.spectra || [];
  if (!r) {
    for (let i = 0; i < spectra.length; i++) {
      const { info = {}, meta = {} } = nmriumState.data?.spectra[i] || {};
      spectra[i] = { ...spectra[i], info, meta };
    }
  }
  // Drop the raw processed spectra (typed arrays, filter history) now that
  // everything needed from them has been copied into `spectra` above —
  // otherwise they stay resident in memory alongside the serialized copy
  // for the rest of the (potentially large, streamed) output write.
  if (nmriumState.data) nmriumState.data.spectra = [];
  const logs = logger.getLogs();
  await outputResult({ nmriumState: { data, version }, images, logs }, o);
}

async function loadSpectrumFromURL(options: RequiredKey<FileOptionsArgs, 'u'>, logger: FifoLogger) {
  const { u: url, include, exclude } = options;

  const { pathname: relativePath, origin: baseURL } = new URL(url);
  const source = {
    entries: [{ relativePath }],
    baseURL,
  };

  const { state } = await core.readFromWebSource(source, { ...getParsingOptions(true), fileFilter: { include, exclude }, logger });

  await processAndSerialize(state, options, logger);
}

async function loadSpectrumFromFilePath(options: RequiredKey<FileOptionsArgs, 'dir'>, logger: FifoLogger) {
  const { dir: path, include, exclude } = options;

  const dirPath = isAbsolute(path) ? path : join(process.cwd(), path);

  const fileCollection = await FileCollection.fromPath(dirPath, {
    unzip: { zipExtensions: ['zip', 'nmredata'] },
    filter: { include, exclude },
  });

  const { state } = await core.read(fileCollection, { ...getParsingOptions(true), logger });

  await processAndSerialize(state, options, logger);
}

async function parseSpectra(argv: yargs.ArgumentsCamelCase<FileOptionsArgs>) {
  const logger = new FifoLogger();
  const { u, dir } = argv;

  try {
    // Branches are mutually exclusive and awaited so a rejection is caught
    // here instead of becoming an unhandled promise rejection, and so -u
    // and --dir can't race to write the same output.
    if (u) {
      await loadSpectrumFromURL({ u, ...argv }, logger);
    } else if (dir) {
      await loadSpectrumFromFilePath({ dir, ...argv }, logger);
    } else {
      throw new Error('Either --u (URL) or --dir (directory) must be provided.');
    }
  } catch (e) {
    logger.error({ stage: 'fatal', details: toMessage(e) }, `Pipeline failed: ${toMessage(e)}`);
    process.stderr.write(`${toMessage(e)}\n`);
    process.exitCode = 1;
  }
}

export { loadSpectrumFromFilePath, loadSpectrumFromURL, parseSpectra };