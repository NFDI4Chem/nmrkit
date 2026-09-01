
import { createWriteStream } from 'node:fs';
import { JsonStreamStringify } from 'json-stream-stringify';

export function outputResult(result: any, outputPath?: string): Promise<void> {
    return new Promise((resolve, reject) => {
        const stream = new JsonStreamStringify(result);

        if (outputPath) {
            const writeStream = createWriteStream(outputPath);
            stream.pipe(writeStream);
            writeStream.on('finish', () => {
                process.stderr.write(`Output written to: ${outputPath}\n`);
                resolve();
            });
            writeStream.on('error', reject);
            stream.on('error', reject);
        } else {
            stream.pipe(process.stdout);
            stream.on('end', () => resolve());
            stream.on('error', reject);
        }
    });
}
