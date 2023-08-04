#!/usr/bin/env node

const yargs = require("yargs");
const loader = require("nmr-load-save");
const fileUtils = require("filelist-utils");

const options = yargs
 .usage("Usage: -u <url>")
 .option("u", { alias: "url", describe: "File URL", type: "string", demandOption: true })
 .argv;

  async function loadSpectrum(url) {
    const {pathname:relativePath,origin:baseURL} = new URL(url);
    const source = {
        entries: [
          {
            relativePath,
          }
        ],
        baseURL
      };
    const fileCollection = await fileUtils.fileCollectionFromWebSource(source,{});
  
    const {
      nmriumState: { data },
    } = await loader.read(fileCollection);
    return data;
  }



 const url = options.u.split(" ");


 loadSpectrum(options.u).then((result)=>{
    console.log(JSON.stringify(result))
 })


