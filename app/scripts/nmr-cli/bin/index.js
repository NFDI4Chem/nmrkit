#!/usr/bin/env node
const {join,isAbsolute}= require("path");
const yargs = require("yargs");
const loader = require("nmr-load-save");
const fileUtils = require("filelist-utils");

const usageMessage ="Usage: nmr-cli -u <url> or -p <path>" 

const options = yargs
 .usage(usageMessage)
 .option("u", { alias: "url", describe: "File URL", type: "string",nargs:1})
 .option("p", { alias: "path", describe: "Directory path", type: "string",nargs:1}).showHelpOnFail();

  async function loadSpectrumFromURL(url) {
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


  async function loadSpectrumFromFilePath(path) {
    const dirPath = isAbsolute(path)?path:join(process.cwd(),path)
  
    const fileCollection = await fileUtils.fileCollectionFromPath(dirPath,{});
  
    const {
      nmriumState: { data },
    } = await loader.read(fileCollection);
    return data;
  }


  const parameters = options.argv;

if(parameters.u && parameters.p){
  options.showHelp();
}else{

  if(parameters.u){
    loadSpectrumFromURL(parameters.u).then((result)=>{
    console.log(JSON.stringify(result))
 })

  }

  if(parameters.p){
    loadSpectrumFromFilePath(parameters.p).then((result)=>{
      console.log(JSON.stringify(result))
   })
  }

}





