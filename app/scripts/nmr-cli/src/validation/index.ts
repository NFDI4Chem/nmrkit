import { readFileSync } from 'node:fs'
import type { CommandModule } from 'yargs'

import { InvalidStructureError } from './molfile.js'
import { createQuickcheckClient, quickcheckUrl, QuickcheckUnavailableError } from './quickcheck.js'
import type { AssignmentSetInput } from './types.js'
import { InvalidInputError, validateAssignments } from './validate.js'

/** Exit codes the API maps to HTTP statuses. */
export const EXIT_INVALID_INPUT = 2
export const EXIT_QUICKCHECK_UNAVAILABLE = 3

function fail(code: number, error: string, message: string): never {
  console.error(JSON.stringify({ error, message }))
  process.exit(code)
}

export const validateAssignmentsCommand: CommandModule = {
  command: ['validate-assignments', 'va'],
  describe: 'Validate 1H/13C assignments against nmrshiftdb2 quickcheck (reads JSON from stdin)',
  handler: async () => {
    let input: AssignmentSetInput
    try {
      input = JSON.parse(readFileSync(0, 'utf-8'))
    } catch (error) {
      fail(EXIT_INVALID_INPUT, 'invalid_input', `Input is not valid JSON: ${String(error)}`)
    }

    try {
      const url = quickcheckUrl()
      const report = await validateAssignments(input, {
        quickcheck: createQuickcheckClient(url),
        url,
      })
      console.log(JSON.stringify(report))
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (error instanceof InvalidInputError || error instanceof InvalidStructureError) {
        fail(EXIT_INVALID_INPUT, 'invalid_input', message)
      }
      if (error instanceof QuickcheckUnavailableError) {
        fail(EXIT_QUICKCHECK_UNAVAILABLE, 'quickcheck_unavailable', message)
      }
      fail(1, 'validation_failed', message)
    }
  },
}
