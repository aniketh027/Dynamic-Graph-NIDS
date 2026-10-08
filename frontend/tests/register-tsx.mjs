// Test-only TSX loader using the already installed TypeScript compiler.
import { registerHooks } from 'node:module'
import { readFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'
import ts from 'typescript'
const root = fileURLToPath(new URL('../src/', import.meta.url))
registerHooks({
  resolve(specifier, context, next) {
    if (specifier.startsWith('@/')) specifier = pathToFileURL(root + specifier.slice(2)).href
    try { return next(specifier,context) } catch(error) {
      if (!specifier.startsWith('.') && !specifier.startsWith('file:')) throw error
      for (const extension of ['.ts','.tsx']) { try { return next(specifier+extension,context) } catch {} }
      throw error
    }
  },
  load(url,context,next) {
    if (/\.tsx?$/.test(url) && !url.includes('/node_modules/')) return {format:'module',shortCircuit:true,
      source:ts.transpileModule(readFileSync(new URL(url),'utf8'),{compilerOptions:{module:ts.ModuleKind.ESNext,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2023}}).outputText}
    return next(url,context)
  },
})
